"""Document indexing and adaptive dense/BM25 retrieval service."""

import hashlib
import re
import uuid
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_agent_template.core.settings import Settings
from ai_agent_template.observability.tracing import agent_span
from ai_agent_template.rag.schemas import RagIngestResponse, RagSource

DocumentType = Literal["knowledge", "voice_sample"]
_NAMESPACE = uuid.UUID("0d2bd8a4-7f4a-4e8c-9f2c-9b9b7c8a1c11")


class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], **kwargs: Any) -> Any: ...
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]: ...


class RagRetriever(Protocol):
    def search(self, query: str) -> list[RagSource]: ...
    def search_voice(self, query: str) -> list[RagSource]: ...


class EmptyRagRetriever:
    def search(self, query: str) -> list[RagSource]:
        return []

    def search_voice(self, query: str) -> list[RagSource]:
        return []


class Reranker(Protocol):
    def rerank(self, query: str, sources: Sequence[RagSource]) -> list[RagSource]: ...


def choose_retrieval_mode(query: str) -> Literal["dense", "sparse", "hybrid"]:
    """Lightweight, auditable query routing. Unknown cases fall back to hybrid."""
    if re.search(r"\b[A-Z]{2,}(?:[-_][A-Za-z0-9]+)*\b|\b\d+\s*(?:eV|Da|m/z)\b", query):
        if not re.search(r"比较|对比|区别|compare|versus|vs\.?|different", query, re.I):
            return "sparse"
    if re.search(r"如何|为什么|原理|机制|解释|how|why|meaning|impact", query, re.I):
        return "dense"
    return "hybrid"


def _chunk_id(source: str, document_type: DocumentType, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_NAMESPACE, f"{document_type}:{source}:{digest}"))


class RagService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        retrieval_stores: dict[str, VectorStore] | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._retrieval_stores = retrieval_stores or {}
        self._reranker = reranker
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def ingest_text(
        self,
        *,
        text: str,
        source: str,
        document_type: DocumentType = "knowledge",
        metadata: dict[str, Any] | None = None,
    ) -> RagIngestResponse:
        docs = self._splitter.create_documents(
            [text],
            metadatas=[{"source": source, "document_type": document_type, **(metadata or {})}],
        )
        return self._index(docs, source, document_type)

    def ingest_sections(
        self,
        *,
        sections: list[Document],
        source: str,
        document_type: DocumentType = "knowledge",
        metadata: dict[str, Any] | None = None,
    ) -> RagIngestResponse:
        """Preserve page/section metadata while indexing child retrieval chunks."""
        docs: list[Document] = []
        for section in sections:
            meta = {
                "source": source,
                "document_type": document_type,
                **(metadata or {}),
                **section.metadata,
            }
            if self._settings.rag_chunk_strategy == "parent_child":
                parent_size = max(self._settings.rag_parent_chunk_size, self._settings.rag_chunk_size)
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)
                for parent in parent_splitter.split_text(section.page_content):
                    for child in self._splitter.split_text(parent):
                        docs.append(Document(
                            page_content=child,
                            metadata={**meta, "parent_excerpt": parent[:parent_size]},
                        ))
            else:
                docs.extend(self._splitter.create_documents([section.page_content], [meta]))
        return self._index(docs, source, document_type)

    def _existing_ids(self, source: str, document_type: DocumentType) -> set[str] | None:
        store = self._vector_store
        client = getattr(store, "client", None)
        collection = getattr(store, "collection_name", None)
        if client is None or collection is None:
            return None
        from qdrant_client import models

        scoped_filter = models.Filter(must=[
            models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
            models.FieldCondition(
                key="metadata.document_type", match=models.MatchValue(value=document_type)
            ),
        ])
        existing: set[str] = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=scoped_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing.update(str(point.id) for point in points)
            if offset is None:
                return existing

    def _index(
        self, documents: list[Document], source: str, document_type: DocumentType
    ) -> RagIngestResponse:
        if not source.strip():
            raise ValueError("Source is required")
        ids = [_chunk_id(source, document_type, doc.page_content + "|" +
                          str(doc.metadata.get("page", "")) + "|" +
                          str(doc.metadata.get("section", ""))) for doc in documents]
        # Avoid duplicate point IDs in one batch while keeping deterministic re-ingest.
        indexed = dict(zip(ids, documents, strict=True))
        with agent_span("rag.ingest", source=source, document_type=document_type) as span:
            old_ids = self._existing_ids(source, document_type)
            pending_ids = set(indexed) - old_ids if old_ids is not None else set(indexed)
            if pending_ids:
                self._vector_store.add_documents(
                    [doc for pid, doc in indexed.items() if pid in pending_ids],
                    ids=[pid for pid in indexed if pid in pending_ids],
                )
            if old_ids is not None:
                stale = old_ids - set(indexed)
                if stale:
                    from qdrant_client import models

                    self._vector_store.client.delete(
                        collection_name=self._vector_store.collection_name,
                        points_selector=models.PointIdsList(points=sorted(stale)),
                        wait=True,
                    )
            span.set_attribute("rag.indexed", len(indexed))
            span.set_attribute("rag.new_embeddings", len(pending_ids))
            return RagIngestResponse(source=source, chunks=len(indexed))

    def delete_source(self, source: str, document_type: DocumentType = "knowledge") -> int:
        ids = self._existing_ids(source, document_type)
        if ids is None:
            raise RuntimeError("Document deletion requires Qdrant-backed storage")
        if ids:
            from qdrant_client import models

            self._vector_store.client.delete(
                collection_name=self._vector_store.collection_name,
                points_selector=models.PointIdsList(points=sorted(ids)),
                wait=True,
            )
        return len(ids)

    def search(self, query: str) -> list[RagSource]:
        return self._search(query=query, k=self._settings.rag_top_k, document_type="knowledge")

    def search_voice(self, query: str) -> list[RagSource]:
        return self._search(
            query=query, k=self._settings.rag_voice_top_k, document_type="voice_sample"
        )

    def _select_store(self, query: str) -> tuple[str, VectorStore]:
        configured = self._settings.rag_retrieval_mode
        mode = choose_retrieval_mode(query) if configured == "adaptive" else configured
        return mode, self._retrieval_stores.get(mode, self._vector_store)

    def _search(
        self, *, query: str, k: int, document_type: DocumentType
    ) -> list[RagSource]:
        fetch_k = max(k * self._settings.rag_search_multiplier, k)
        mode, store = self._select_store(query)
        with agent_span("rag.search", document_type=document_type, retrieval_mode=mode) as span:
            results = store.similarity_search_with_score(query, k=fetch_k)
            candidates: list[RagSource] = []
            for document, score in results:
                metadata = dict(document.metadata)
                if metadata.get("document_type", "knowledge") != document_type:
                    continue
                if (
                    self._settings.rag_score_threshold is not None
                    and mode == "dense"
                    and score < self._settings.rag_score_threshold
                ):
                    continue
                source = str(metadata.pop("source", "unknown"))
                candidates.append(
                    RagSource(
                        source=source, content=document.page_content, score=float(score),
                        metadata=metadata,
                    )
                )
            if self._reranker and candidates:
                candidates = self._reranker.rerank(query, candidates)
            span.set_attribute("rag.results", min(k, len(candidates)))
            return candidates[:k]

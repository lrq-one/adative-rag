"""Document indexing and adaptive dense/BM25 retrieval service."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_agent_template.core.settings import Settings
from ai_agent_template.observability.tracing import agent_span
from ai_agent_template.rag.indexing import VersionedIndexer
from ai_agent_template.rag.repository import EvidenceRepository, digest
from ai_agent_template.rag.schemas import EvidenceRecord, IndexStats, RagIngestResponse, RagSource

DocumentType = Literal["knowledge", "voice_sample"]
RetrievalMode = Literal["dense", "sparse", "hybrid"]


class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], **kwargs: Any) -> Any: ...
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]: ...


class RagRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> list[RagSource]: ...
    def search_voice(self, query: str) -> list[RagSource]: ...


class EmptyRagRetriever:
    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> list[RagSource]:
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


class RagService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        retrieval_stores: Mapping[str, VectorStore] | None = None,
        repository: EvidenceRepository | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._retrieval_stores = retrieval_stores or {}
        self._reranker = reranker
        self._temporary_storage = None
        data_path = settings.rag_data_path
        if repository is None and not getattr(vector_store, "client", None) and data_path == ".rag":
            self._temporary_storage = TemporaryDirectory()
            data_path = self._temporary_storage.name
        self.repository = repository or EvidenceRepository(Path(data_path) / "manifest.sqlite3")
        self.indexer = VersionedIndexer(settings, vector_store, self.repository)
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
        document_id: str | None = None,
        namespace: str = "default",
    ) -> RagIngestResponse:
        return self.indexer.ingest(
            [Document(page_content=text, metadata=metadata or {})],
            source,
            document_type,
            document_id=document_id,
            namespace=namespace,
        )

    def ingest_sections(
        self,
        *,
        sections: list[Document],
        source: str,
        document_type: DocumentType = "knowledge",
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
        namespace: str = "default",
        file_hash: str | None = None,
        original_file: str | None = None,
        preprocessing_stats: IndexStats | None = None,
        parser_version: str = "scientific-parser-v3",
    ) -> RagIngestResponse:
        """Preserve page/section metadata while indexing child retrieval chunks."""
        return self.indexer.ingest(
            sections,
            source,
            document_type,
            metadata,
            document_id=document_id,
            namespace=namespace,
            file_hash=file_hash,
            original_file=original_file,
            preprocessing_stats=preprocessing_stats,
            parser_version=parser_version,
        )

    def _existing_ids(self, source: str, document_type: DocumentType) -> set[str] | None:
        store = self._vector_store
        client = getattr(store, "client", None)
        collection = getattr(store, "collection_name", None)
        if client is None or collection is None:
            return None
        from qdrant_client import models

        scoped_filter = models.Filter(
            must=[
                models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source)),
                models.FieldCondition(
                    key="metadata.document_type", match=models.MatchValue(value=document_type)
                ),
            ]
        )
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

    def delete_source(self, source: str, document_type: DocumentType = "knowledge") -> int:
        managed = [
            item
            for item in self.repository.snapshot(self._settings.rag_namespace)
            if (item.source_filename == source or item.document_id == source)
            and item.document_type == document_type
        ]
        if managed:
            count = 0
            from qdrant_client import models

            for item in managed:
                with self.repository.document_lock(item.document_id):
                    count += self.repository.retire(item.document_id, lock_held=True)
                    self.indexer.cache.remove_owner(item.document_id)
                    client = getattr(self._vector_store, "client", None)
                    if client:
                        client.delete(
                            collection_name=getattr(self._vector_store, "collection_name"),
                            points_selector=models.FilterSelector(
                                filter=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="metadata.document_id",
                                            match=models.MatchValue(value=item.document_id),
                                        )
                                    ]
                                )
                            ),
                            wait=True,
                        )
            return count
        ids = self._existing_ids(source, document_type)
        if ids is None:
            raise RuntimeError("Document deletion requires Qdrant-backed storage")
        if ids:
            from qdrant_client import models

            getattr(self._vector_store, "client").delete(
                collection_name=getattr(self._vector_store, "collection_name"),
                points_selector=models.PointIdsList(points=sorted(ids)),
                wait=True,
            )
        return len(ids)

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> list[RagSource]:
        return self._search(
            query=query,
            k=self._settings.rag_top_k,
            document_type="knowledge",
            route=mode,
            source_ids=source_ids,
        )

    def search_voice(self, query: str) -> list[RagSource]:
        return self._search(
            query=query, k=self._settings.rag_voice_top_k, document_type="voice_sample"
        )

    def _select_store(
        self,
        query: str,
        route: RetrievalMode | None = None,
    ) -> tuple[str, VectorStore]:
        configured = self._settings.rag_retrieval_mode
        # The graph's semantic plan takes precedence. Direct search retains its offline fallback.
        mode = route or (choose_retrieval_mode(query) if configured == "adaptive" else configured)
        if self._retrieval_stores and mode not in self._retrieval_stores:
            raise ValueError(f"Retrieval mode {mode!r} is not configured")
        return mode, self._retrieval_stores.get(mode, self._vector_store)

    def _search(
        self,
        *,
        query: str,
        k: int,
        document_type: DocumentType,
        route: RetrievalMode | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> list[RagSource]:
        fetch_k = max(k * self._settings.rag_search_multiplier, k)
        mode, store = self._select_store(query, route)
        snapshot = self.repository.snapshot(self._settings.rag_namespace)
        snapshot = [
            item
            for item in snapshot
            if item.document_type == document_type
            and (
                not source_ids
                or item.document_id in source_ids
                or item.source_filename in source_ids
            )
        ]
        active_versions = {(item.document_id, item.document_version) for item in snapshot}
        active_evidence = {
            (item.document_id, point): evidence
            for item in snapshot
            for point, evidence in zip(
                item.chunk_ids, item.evidence_ids or item.chunk_ids, strict=True
            )
        }
        managed_sources = self.repository.managed_sources()
        with agent_span("rag.search", document_type=document_type, retrieval_mode=mode) as span:
            search_kwargs: dict[str, Any] = {}
            if source_ids and getattr(store, "supports_source_ids", False):
                search_kwargs["source_ids"] = list(source_ids)
            if getattr(store, "client", None) is not None:
                from qdrant_client import models

                conditions: list[models.FieldCondition | models.Filter] = [
                    models.FieldCondition(
                        key="metadata.document_type",
                        match=models.MatchValue(value=document_type),
                    )
                ]
                legacy = models.Filter(
                    must=[
                        models.IsEmptyCondition(
                            is_empty=models.PayloadField(key="metadata.document_id")
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="metadata.source", match=models.MatchAny(any=managed_sources)
                        )
                    ]
                    if managed_sources
                    else None,
                )
                if source_ids:
                    legacy.must = [
                        models.IsEmptyCondition(
                            is_empty=models.PayloadField(key="metadata.document_id")
                        ),
                        models.FieldCondition(
                            key="metadata.source", match=models.MatchAny(any=list(source_ids))
                        ),
                    ]
                allowed = [
                    legacy,
                    *(
                        self.indexer.version_filter(doc, version)
                        for doc, version in active_versions
                    ),
                ]
                conditions.append(models.Filter(should=allowed))
                search_kwargs["filter"] = models.Filter(must=conditions)
            results = store.similarity_search_with_score(query, k=fetch_k, **search_kwargs)
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
                if metadata.get("document_id"):
                    evidence_id = active_evidence.get(
                        (
                            metadata["document_id"],
                            str(metadata.get("chunk_id") or metadata.get("_id")),
                        )
                    )
                    if not evidence_id:
                        continue
                    resolved = self.repository.resolve(evidence_id)
                    if not resolved:
                        continue
                    evidence = resolved[0]
                    source = evidence.source_filename
                    metadata = dict(evidence.metadata)
                    document = Document(page_content=evidence.text, metadata=metadata)
                else:
                    if source in managed_sources or (source_ids and source not in source_ids):
                        continue
                    legacy_id = str(
                        metadata.get("_id")
                        or digest(source + document.page_content + str(metadata))
                    )
                    evidence = EvidenceRecord(
                        evidence_id=legacy_id,
                        document_id=digest(source),
                        document_version=0,
                        chunk_id=legacy_id,
                        source_filename=source,
                        page_number=metadata.get("page"),
                        section_title=metadata.get("section"),
                        content_type=metadata.get("content_type") or "text",
                        text=document.page_content,
                        text_hash=digest(document.page_content),
                        metadata={**metadata, "parser_status": "legacy_unversioned"},
                    )
                candidates.append(
                    RagSource(
                        source=source,
                        content=document.page_content,
                        score=float(score),
                        metadata=metadata,
                        evidence=evidence,
                    )
                )
            if self._reranker and candidates:
                candidates = self._reranker.rerank(query, candidates)
            span.set_attribute("rag.results", min(k, len(candidates)))
            return candidates[:k]

    def evidence_status(self, record: EvidenceRecord) -> str:
        if record.document_version == 0:
            return (
                "legacy"
                if record.source_filename not in self.repository.managed_sources()
                else "unavailable"
            )
        resolved = self.repository.resolve(record.evidence_id)
        return resolved[1] if resolved else "unavailable"

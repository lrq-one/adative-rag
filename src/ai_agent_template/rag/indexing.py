"""Stage complete immutable versions, then publish via a transactional manifest."""

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import PersistentCache, embedding_identity
from ai_agent_template.rag.loaders import ocr_identity
from ai_agent_template.rag.repository import EvidenceRepository, digest, fingerprint, timestamp
from ai_agent_template.rag.schemas import (
    DocumentManifest,
    EvidenceRecord,
    IndexStats,
    RagIngestResponse,
)

ID_NAMESPACE = uuid.UUID("0d2bd8a4-7f4a-4e8c-9f2c-9b9b7c8a1c11")


def logical_id(namespace: str, source: str, kind: str, document_id: str | None = None) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{namespace}:{kind}:{document_id or source}"))


class VersionedIndexer:
    def __init__(self, settings: Settings, store: Any, repository: EvidenceRepository) -> None:
        self.settings, self.store, self.repository = settings, store, repository
        self.cache = PersistentCache(repository.path.parent / "cache.sqlite3")

    def model_identity(self) -> str:
        return str(
            getattr(
                getattr(self.store, "embeddings", None),
                "model_id",
                embedding_identity(self.settings),
            )
        )

    def index_identity(self) -> str:
        return fingerprint(
            [
                "evidence-v2",
                self.model_identity(),
                getattr(
                    getattr(self.store, "sparse_embeddings", None),
                    "model_id",
                    self.settings.rag_sparse_model_name,
                ),
                getattr(self.store, "collection_name", None),
            ]
        )

    def complete(self, manifest: DocumentManifest) -> bool:
        client = getattr(self.store, "client", None)
        return client is None or client.count(
            collection_name=self.store.collection_name,
            count_filter=self.version_filter(manifest.document_id, manifest.document_version),
            exact=True,
        ).count == len(manifest.chunk_ids)

    def ingest(
        self,
        sections: Sequence[Document],
        source: str,
        kind: str = "knowledge",
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
        namespace: str = "default",
        file_hash: str | None = None,
        original_file: str | None = None,
        parser_version: str = "scientific-parser-v3",
        preprocessing_stats: IndexStats | None = None,
    ) -> RagIngestResponse:
        started = time.perf_counter()
        if not source.strip() or not sections:
            raise ValueError("Nonempty source and content are required")
        # Persisted UUIDs may be used directly for updates; namespace ownership is checked.
        doc_id = (
            document_id
            if document_id and self.repository.versions(document_id)
            else logical_id(namespace, source, kind, document_id)
        )
        with self.repository.document_lock(doc_id), self.cache.bind(doc_id):
            versions = self.repository.versions(doc_id)
            for abandoned in versions:
                if abandoned.index_status == "building":
                    abandoned.index_status = "failed"
                    abandoned.stats.update_status = "interrupted"
                    self.repository.save(abandoned)
            if versions and (
                versions[-1].namespace != namespace or versions[-1].document_type != kind
            ):
                raise ValueError(
                    "Document identity belongs to a different namespace or document type"
                )
            old = self.repository.active(doc_id)
            content_hash = fingerprint([section.page_content for section in sections])
            input_hash = fingerprint(
                [(section.page_content, section.metadata) for section in sections]
                + [metadata or {}, source, file_hash]
            )
            config_hash = fingerprint(
                {
                    "strategy": self.settings.rag_chunk_strategy,
                    "size": self.settings.rag_chunk_size,
                    "overlap": self.settings.rag_chunk_overlap,
                    "parent_size": self.settings.rag_parent_chunk_size,
                    "version": "chunker-v3",
                }
            )
            compatibility = (
                input_hash,
                config_hash,
                self.model_identity(),
                parser_version,
                self.index_identity(),
            )
            if (
                old
                and compatibility
                == (
                    old.input_fingerprint or old.file_sha256,
                    old.chunker_config_hash,
                    old.embedding_model_id,
                    old.parser_version,
                    old.index_fingerprint,
                )
                and self.complete(old)
            ):
                stats = old.stats.model_copy(
                    update={
                        "unchanged_chunks": len(old.chunk_ids),
                        "new_chunks": 0,
                        "modified_chunks": 0,
                        "deleted_chunks": 0,
                        "metadata_only_updates": 0,
                        "vector_reuses": 0,
                        "embedding_cache_hits": 0,
                        "embedding_cache_misses": 0,
                        "sparse_cache_hits": 0,
                        "sparse_cache_misses": 0,
                        "chunk_cache_hits": 0,
                        "page_text_cache_hits": preprocessing_stats.page_text_cache_hits
                        if preprocessing_stats
                        else 0,
                        "parse_cache_hits": preprocessing_stats.parse_cache_hits
                        if preprocessing_stats
                        else 0,
                        "ocr_cache_hits": preprocessing_stats.ocr_cache_hits
                        if preprocessing_stats
                        else 0,
                        "vlm_cache_hits": preprocessing_stats.vlm_cache_hits
                        if preprocessing_stats
                        else 0,
                        "parsing_time_ms": preprocessing_stats.parsing_time_ms
                        if preprocessing_stats
                        else 0,
                        "chunking_time_ms": 0,
                        "embedding_time_ms": 0,
                        "qdrant_update_time_ms": 0,
                        "update_status": "unchanged",
                        "total_index_time_ms": (time.perf_counter() - started) * 1000
                        + (preprocessing_stats.parsing_time_ms if preprocessing_stats else 0),
                    }
                )
                result = RagIngestResponse(
                    source=source,
                    chunks=len(old.chunk_ids),
                    document_id=doc_id,
                    document_version=old.document_version,
                    stats=stats,
                )
                self.repository.record_job(result)
                return result
            version = max((item.document_version for item in versions), default=0) + 1
            manifest = DocumentManifest(
                document_id=doc_id,
                document_version=version,
                namespace=namespace,
                source_filename=source,
                document_type=kind,
                file_sha256=file_hash or input_hash,
                input_fingerprint=input_hash,
                normalized_content_hash=content_hash,
                parser_version=parser_version,
                chunker_config_hash=config_hash,
                embedding_model_id=self.model_identity(),
                index_fingerprint=self.index_identity(),
                created_at=timestamp(),
                original_file=original_file,
                stats=(preprocessing_stats or IndexStats()).model_copy(
                    update={"index_version": version}
                ),
                ocr_model_version=ocr_identity() if self.settings.rag_ocr_enabled else None,
                vlm_model_version=self.settings.rag_vlm_model
                if self.settings.rag_vlm_enabled
                else None,
            )
            self.repository.save(manifest)
            try:
                chunk_start = time.perf_counter()
                docs, records = self.chunk(sections, manifest, metadata or {})
                reused = {}
                if old and old.index_fingerprint == manifest.index_fingerprint:
                    available: defaultdict[str, deque[EvidenceRecord]] = defaultdict(deque)
                    for evidence_id in old.evidence_ids or old.chunk_ids:
                        resolved = self.repository.resolve(evidence_id)
                        if resolved:
                            available[resolved[0].text_hash].append(resolved[0])
                    for doc, record in zip(docs, records, strict=True):
                        matches = available[record.text_hash]
                        if matches:
                            previous = matches.popleft()
                            record.chunk_id = previous.chunk_id
                            doc.metadata["chunk_id"] = record.chunk_id
                            record.metadata["chunk_id"] = record.chunk_id
                            reused[record.chunk_id] = record.evidence_id
                manifest.chunk_ids = [record.chunk_id for record in records]
                manifest.evidence_ids = [record.evidence_id for record in records]
                manifest.stats.total_chunks = len(docs)
                manifest.stats.new_chunks = len(docs)
                if old:
                    self.diff(old, records, manifest.stats)
                manifest.stats.chunking_time_ms = (time.perf_counter() - chunk_start) * 1000
                self.repository.save(manifest, records)
                write_start = time.perf_counter()
                dense = getattr(self.store, "embeddings", None)
                sparse = getattr(self.store, "sparse_embeddings", None)

                def usage(encoder: Any) -> tuple[int, int, float]:
                    return encoder.usage_for(doc_id) if hasattr(encoder, "usage_for") else (0, 0, 0)

                before = [usage(encoder) for encoder in (dense, sparse)]
                client = getattr(self.store, "client", None)
                existing = (
                    {
                        str(point.id): point
                        for point in client.retrieve(
                            collection_name=self.store.collection_name,
                            ids=list(reused),
                            with_payload=True,
                            with_vectors=False,
                        )
                    }
                    if client and reused
                    else {}
                )
                fresh = [
                    (doc, record)
                    for doc, record in zip(docs, records, strict=True)
                    if record.chunk_id not in existing
                ]
                if fresh:
                    self.store.add_documents(
                        [doc for doc, record in fresh],
                        ids=[record.chunk_id for doc, record in fresh],
                    )
                if existing:
                    assert client is not None
                    groups = defaultdict(list)
                    for point_id, point in existing.items():
                        prior_meta = (point.payload or {}).get("metadata", {})
                        versions_for_point = tuple(
                            sorted(
                                set(
                                    [
                                        *prior_meta.get(
                                            "versions", [prior_meta.get("document_version", 1)]
                                        ),
                                        version,
                                    ]
                                )
                            )
                        )
                        groups[versions_for_point].append(point_id)
                    for versions_for_point, point_ids in groups.items():
                        client.set_payload(
                            collection_name=self.store.collection_name,
                            payload={"versions": list(versions_for_point)},
                            key="metadata",
                            points=point_ids,
                            wait=True,
                        )
                manifest.stats.vector_reuses = len(existing)
                for encoder, prior, hit_field, miss_field in zip(
                    (dense, sparse),
                    before,
                    ("embedding_cache_hits", "sparse_cache_hits"),
                    ("embedding_cache_misses", "sparse_cache_misses"),
                    strict=True,
                ):
                    current = usage(encoder)
                    setattr(manifest.stats, hit_field, current[0] - prior[0])
                    setattr(manifest.stats, miss_field, current[1] - prior[1])
                manifest.stats.embedding_time_ms = usage(dense)[2] - before[0][2]
                client = getattr(self.store, "client", None)
                if client:
                    count = client.count(
                        collection_name=self.store.collection_name,
                        count_filter=self.version_filter(doc_id, version),
                        exact=True,
                    ).count
                    if count != len(docs):
                        raise RuntimeError(
                            "Incomplete staged index; previous version remains active"
                        )
                    points = client.retrieve(
                        collection_name=self.store.collection_name,
                        ids=manifest.chunk_ids,
                        with_vectors=True,
                        with_payload=True,
                    )
                    if any(
                        not isinstance(point.vector, dict)
                        or not {"dense", "sparse"}.issubset(point.vector)
                        for point in points
                    ):
                        raise RuntimeError("Staged dense/sparse vectors are incomplete")
                manifest.stats.qdrant_update_time_ms = (time.perf_counter() - write_start) * 1000
                manifest.stats.total_index_time_ms = (
                    time.perf_counter() - started
                ) * 1000 + manifest.stats.parsing_time_ms
                self.repository.save(manifest)
                self.repository.publish(manifest)
            except Exception as error:
                manifest.index_status = "failed"
                manifest.stats.update_status = "failed"
                manifest.stats.error_type = type(error).__name__
                manifest.stats.total_index_time_ms = (
                    time.perf_counter() - started
                ) * 1000 + manifest.stats.parsing_time_ms
                self.repository.save(manifest)
                self.repository.record_job(
                    RagIngestResponse(
                        source=source,
                        chunks=manifest.stats.total_chunks,
                        document_id=doc_id,
                        document_version=version,
                        stats=manifest.stats,
                    )
                )
                raise
            result = RagIngestResponse(
                source=source,
                chunks=len(docs),
                document_id=doc_id,
                document_version=version,
                stats=manifest.stats,
            )
            self.repository.record_job(result)
            return result

    def activate(self, document_id: str, version: int, namespace: str) -> DocumentManifest:
        with self.repository.document_lock(document_id):
            manifest = next(
                (
                    item
                    for item in self.repository.versions(document_id)
                    if item.document_version == version
                    and item.namespace == namespace
                    and item.index_status in {"active", "historical"}
                ),
                None,
            )
            if manifest is None:
                raise ValueError("Historical version is unavailable")
            if manifest.embedding_model_id != self.model_identity():
                raise ValueError("Historical version belongs to a different embedding model")
            if manifest.index_fingerprint != self.index_identity() or not self.complete(manifest):
                raise ValueError(
                    "Historical index configuration or point versions are incompatible"
                )
            client = getattr(self.store, "client", None)
            if client is None:
                raise ValueError("Version activation requires Qdrant-backed storage")
            points = client.retrieve(
                collection_name=self.store.collection_name,
                ids=manifest.chunk_ids,
                with_vectors=True,
            )
            if len(points) != len(manifest.chunk_ids) or any(
                not isinstance(point.vector, dict) or not {"dense", "sparse"}.issubset(point.vector)
                for point in points
            ):
                raise ValueError("Historical index is incomplete")
            self.repository.publish(manifest)
            return manifest

    @staticmethod
    def version_filter(document_id: str, version: int) -> models.Filter:
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="metadata.versions", match=models.MatchValue(value=version)
                ),
            ]
        )

    def chunk(
        self, sections: Sequence[Document], manifest: DocumentManifest, metadata: dict[str, Any]
    ) -> tuple[list[Document], list[EvidenceRecord]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.rag_chunk_size, chunk_overlap=self.settings.rag_chunk_overlap
        )
        docs, records = [], []
        if self.settings.rag_chunk_strategy == "recursive":
            spans, offset = [], 0
            for original in sections:
                if original.metadata.get("dataset_unit_id"):
                    spans.append(
                        {
                            "unit_id": original.metadata["dataset_unit_id"],
                            "start": offset,
                            "end": offset + len(original.page_content),
                        }
                    )
                offset += len(original.page_content) + 2
            sections = [
                Document(
                    page_content="\n\n".join(section.page_content for section in sections),
                    metadata={
                        "parser_status": "merged_sections_location_unavailable",
                        "dataset_spans": spans,
                    },
                )
            ]
        for section_index, section in enumerate(sections):
            parents = [section.page_content]
            cache_key = fingerprint([section.page_content, manifest.chunker_config_hash])
            cached = self.cache.get("chunks", cache_key)
            if cached is not None:
                manifest.stats.chunk_cache_hits += 1
            if self.settings.rag_chunk_strategy == "parent_child":
                import re

                units = [
                    unit for unit in re.split(r"\n\s*\n", section.page_content) if unit.strip()
                ]
                parents = [
                    parent
                    for unit in units
                    for parent in RecursiveCharacterTextSplitter(
                        chunk_size=max(
                            self.settings.rag_parent_chunk_size, self.settings.rag_chunk_size
                        ),
                        chunk_overlap=0,
                    ).split_text(unit)
                ]
            specifications = (
                cached
                if cached is not None
                else [(parent, splitter.split_text(parent)) for parent in parents]
            )
            if cached is None:
                self.cache.put_many("chunks", {cache_key: specifications})
            parent_cursor = 0
            for parent_index, (parent, children) in enumerate(specifications):
                parent_start = section.page_content.find(parent, parent_cursor)
                parent_cursor = max(parent_start + len(parent), parent_cursor)
                child_cursor = 0
                parent_id = str(
                    uuid.uuid5(
                        ID_NAMESPACE,
                        f"{manifest.document_id}:{manifest.document_version}:p:{section_index}:{parent_index}",
                    )
                )
                for child_index, child in enumerate(children):
                    child_start = parent.find(child, child_cursor)
                    child_cursor = max(child_start + 1, child_cursor)
                    chunk_id = str(
                        uuid.uuid5(
                            ID_NAMESPACE,
                            f"{manifest.document_id}:{manifest.document_version}:c:{section_index}:{parent_index}:{child_index}",
                        )
                    )
                    meta = {
                        **metadata,
                        **section.metadata,
                        "source": manifest.source_filename,
                        "document_type": manifest.document_type,
                        "namespace": manifest.namespace,
                        "document_id": manifest.document_id,
                        "document_version": manifest.document_version,
                        "versions": [manifest.document_version],
                        "chunk_id": chunk_id,
                        "evidence_id": chunk_id,
                        "parent_chunk_id": parent_id,
                        "text_hash": digest(child),
                        "unit_index": section_index,
                        "child_index": child_index,
                        "parent_index": parent_index,
                        "parser_status": section.metadata.get(
                            "parser_status", "text_extracted_bbox_unavailable"
                        ),
                    }
                    if self.settings.rag_chunk_strategy == "parent_child":
                        meta["parent_excerpt"] = parent
                    if parent_start >= 0 and child_start >= 0:
                        meta["chunk_start"] = parent_start + child_start
                        meta["chunk_end"] = meta["chunk_start"] + len(child)
                        meta["dataset_spans"] = section.metadata.get("dataset_spans", [])
                        if section.metadata.get("dataset_unit_id"):
                            meta["dataset_spans"] = [
                                {
                                    "unit_id": section.metadata["dataset_unit_id"],
                                    "start": 0,
                                    "end": len(section.page_content),
                                }
                            ]
                    record = EvidenceRecord(
                        evidence_id=chunk_id,
                        document_id=manifest.document_id,
                        document_version=manifest.document_version,
                        chunk_id=chunk_id,
                        parent_chunk_id=parent_id,
                        source_filename=manifest.source_filename,
                        page_number=meta.get("page"),
                        section_title=meta.get("section"),
                        content_type=meta.get("content_type") or "text",
                        text=child,
                        text_hash=digest(child),
                        bbox=meta.get("bbox"),
                        metadata=meta,
                    )
                    docs.append(Document(page_content=child, metadata=meta))
                    records.append(record)
        return docs, records

    def diff(self, old: DocumentManifest, records: list[EvidenceRecord], stats: IndexStats) -> None:
        by_hash: defaultdict[str, deque[EvidenceRecord]] = defaultdict(deque)
        for evidence_id in old.evidence_ids or old.chunk_ids:
            resolved = self.repository.resolve(evidence_id)
            if resolved:
                by_hash[resolved[0].text_hash].append(resolved[0])
        added = 0
        for record in records:
            matches = by_hash[record.text_hash]
            if matches:
                previous = matches.popleft()

                def location(item: EvidenceRecord) -> tuple[Any, ...]:
                    return (
                        item.page_number,
                        item.section_title,
                        item.metadata.get("unit_index"),
                        item.metadata.get("parent_index"),
                        item.metadata.get("child_index"),
                        item.source_filename,
                    )

                if location(previous) == location(record):
                    stats.unchanged_chunks += 1
                else:
                    stats.metadata_only_updates += 1
            else:
                added += 1
        deleted = sum(len(items) for items in by_hash.values())
        stats.modified_chunks = min(added, deleted)
        stats.new_chunks = added - stats.modified_chunks
        stats.deleted_chunks = deleted - stats.modified_chunks

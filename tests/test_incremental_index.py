from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient, models

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import CachedEmbeddings, CachedSparseEmbeddings, PersistentCache
from ai_agent_template.rag.loaders import load_upload_documents
from ai_agent_template.rag.schemas import IndexStats
from ai_agent_template.rag.service import RagService


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[float(len(text) % 3 + 1), 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 1.0]


class Sparse(SparseEmbeddings):
    def embed_documents(self, texts):
        return [SparseVector(indices=[1], values=[float(len(text))]) for text in texts]

    def embed_query(self, text):
        return SparseVector(indices=[1], values=[1.0])


def make_service(tmp_path, client=None, inner=None, model="toy-v1"):
    client = client or QdrantClient(":memory:")
    if not client.collection_exists("evidence"):
        client.create_collection(
            "evidence",
            vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    cache = PersistentCache(tmp_path / "cache.sqlite3")
    inner = inner or CountingEmbeddings()
    dense = CachedEmbeddings(inner, cache, model, 2)
    sparse = CachedSparseEmbeddings(Sparse(), cache, "sparse-toy-v1")
    common = dict(
        client=client,
        collection_name="evidence",
        embedding=dense,
        sparse_embedding=sparse,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    stores = {
        mode: QdrantVectorStore(**common, retrieval_mode=getattr(RetrievalMode, mode.upper()))
        for mode in ("dense", "sparse", "hybrid")
    }
    inner.texts.clear()  # Exclude the adapter's startup dimension-validation probe.
    settings = Settings(
        rag_data_path=str(tmp_path),
        rag_embedding_model_name=model,
        rag_embedding_dimensions=2,
        rag_chunk_overlap=0,
        rag_chunk_size=100,
        rag_parent_chunk_size=200,
    )
    return RagService(settings, stores["hybrid"], retrieval_stores=stores), inner, client


def test_duplicate_and_local_edit_reuse_actual_vectors(tmp_path):
    service, inner, client = make_service(tmp_path)
    original = [
        Document(
            page_content=f"Section {i} reports a distinct measurement.",
            metadata={"section": str(i)},
        )
        for i in range(10)
    ]
    first = service.ingest_sections(sections=original, source="paper.md")
    assert len(inner.texts) == 10
    same = service.ingest_sections(sections=original, source="paper.md")
    assert same.stats.update_status == "unchanged" and len(inner.texts) == 10
    edited = [
        *original[:4],
        Document(page_content="Changed method and results.", metadata={"section": "4"}),
        *original[5:],
    ]
    updated = service.ingest_sections(sections=edited, source="paper.md")
    assert len(inner.texts) == 11 and updated.stats.embedding_cache_misses == 1
    assert updated.stats.embedding_cache_hits == 0 and updated.stats.chunk_cache_hits == 9
    assert updated.stats.vector_reuses == 9
    assert updated.stats.modified_chunks == 1 and updated.stats.unchanged_chunks == 9
    for mode in ("dense", "sparse", "hybrid"):
        results = service.search("measurement", mode=mode)
        assert results and all(source.evidence.document_version == 2 for source in results)
    assert (
        service.repository.resolve(service.repository.versions(first.document_id)[0].chunk_ids[0])[
            1
        ]
        == "historical"
    )
    client.close()


def test_duplicate_text_different_positions_and_documents_share_only_embedding(tmp_path):
    service, inner, client = make_service(tmp_path)
    first = service.ingest_sections(
        sections=[
            Document(page_content="Same.", metadata={"page": 1}),
            Document(page_content="Same.", metadata={"page": 2}),
        ],
        source="paper.pdf",
    )
    other = service.ingest_text(text="Same.", source="other.pdf")
    assert len(inner.texts) == 1 and first.document_id != other.document_id
    assert len(set(service.repository.versions(first.document_id)[0].chunk_ids)) == 2
    service.delete_source(first.document_id)
    assert service.search("Same.") and all(
        source.source == "other.pdf" for source in service.search("Same.")
    )
    assert service.indexer.cache.get_many("dense", []) == {}
    client.close()


def test_page_metadata_change_and_restart(tmp_path):
    service, inner, client = make_service(tmp_path)
    service.ingest_sections(
        sections=[Document(page_content="Stable text.", metadata={"page": 1})], source="paper.pdf"
    )
    again, restarted_inner, _ = make_service(tmp_path, client)
    result = again.ingest_sections(
        sections=[Document(page_content="Stable text.", metadata={"page": 8})], source="paper.pdf"
    )
    assert restarted_inner.texts == [] and result.stats.metadata_only_updates == 1
    assert again.search("Stable")[0].evidence.page_number == 8
    changed, changed_inner, _ = make_service(tmp_path, client, model="toy-v2")
    changed.ingest_sections(
        sections=[Document(page_content="Stable text.", metadata={"page": 8})], source="paper.pdf"
    )
    assert len(changed_inner.texts) == 1
    client.close()


def test_prepend_delete_and_rename_keep_points_and_current_locations(tmp_path):
    service, inner, client = make_service(tmp_path)
    original = "First independent paragraph.\n\nSecond independent paragraph.\n\nThird independent paragraph."
    first = service.ingest_text(text=original, source="old.md")
    old = service.repository.active(first.document_id)
    result = service.ingest_text(
        text="New introductory paragraph.\n\n" + original,
        source="new.md",
        document_id=first.document_id,
    )
    assert result.stats.vector_reuses == 3 and result.stats.embedding_cache_misses == 1
    hits = service.search("paragraph", source_ids=["new.md"])
    assert hits and all(
        hit.source == "new.md" and hit.evidence.document_version == 2 for hit in hits
    )
    assert service.search("paragraph", source_ids=["old.md"]) == []
    service.ingest_text(
        text="Second independent paragraph.", source="new.md", document_id=first.document_id
    )
    hits = service.search("paragraph")
    assert hits and all(
        hit.content == "Second independent paragraph." and hit.evidence.document_version == 3
        for hit in hits
    )
    assert service.repository.resolve(old.evidence_ids[0])[1] == "historical"
    client.close()


def test_failed_staging_and_interrupted_manifest_preserve_active_version(tmp_path, monkeypatch):
    service, inner, client = make_service(tmp_path)
    first = service.ingest_text(text="Active content.", source="paper.md")
    old = service.repository.active(first.document_id)
    with monkeypatch.context() as patch:
        patch.setattr(
            service.indexer.store,
            "add_documents",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
        with pytest.raises(RuntimeError):
            service.ingest_text(text="Failed new content.", source="paper.md")
    assert service.search("content")[0].content == "Active content."
    abandoned = old.model_copy(update={"document_version": 3, "index_status": "building"})
    service.repository.save(abandoned)
    resumed = service.ingest_text(text="Recovered content.", source="paper.md")
    assert resumed.document_version == 4
    assert service.repository.versions(first.document_id)[2].stats.update_status == "interrupted"
    assert service.search("content")[0].content == "Recovered content."
    service.indexer.activate(first.document_id, 1, "default")
    assert service.search("content")[0].content == "Active content."
    assert service.repository.jobs()[1]["stats"]["update_status"] == "failed"
    client.close()


def test_concurrent_same_document_updates_are_serialized(tmp_path):
    service, inner, client = make_service(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: service.ingest_text(text="Same concurrent content.", source="paper.md"),
                range(2),
            )
        )
    assert {result.document_version for result in results} == {1}
    assert len(inner.texts) == 1
    client.close()


async def test_persisted_parser_cache_survives_restart(tmp_path):
    settings = Settings(rag_data_path=str(tmp_path))
    cache = PersistentCache(tmp_path / "cache.sqlite3")
    for expected_hits in (0, 1):
        stats = IndexStats()
        docs = await load_upload_documents(
            UploadFile(file=BytesIO(b"# Methods\n\nMeasured data."), filename="paper.md"),
            settings,
            cache,
            stats,
        )
        assert docs and stats.parse_cache_hits == expected_hits
        cache = PersistentCache(tmp_path / "cache.sqlite3")


def test_cache_corrupt_dimension_is_rejected(tmp_path):
    cache = PersistentCache(tmp_path / "cache.sqlite3")
    encoder = CachedEmbeddings(CountingEmbeddings(), cache, "toy-v1", 2)
    encoder.embed_documents(["text"])
    from ai_agent_template.rag.repository import digest, fingerprint

    key = fingerprint(["toy-v1", digest("text")])
    cache.put_many("dense", {key: [1.0]})
    with pytest.raises(ValueError, match="dimension"):
        encoder.embed_documents(["text"])


def test_artifact_content_changes_invalidate_identity(tmp_path):
    from types import SimpleNamespace

    from ai_agent_template.rag.cache import resolved_embedding_identity

    model = tmp_path / "model.onnx"
    model.write_bytes(b"weight-version-one")
    encoder = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(_model_dir=tmp_path)))
    before = resolved_embedding_identity(Settings(), encoder)
    model.write_bytes(b"different-weight-version-two")
    assert resolved_embedding_identity(Settings(), encoder) != before


def test_sparse_identity_change_rebuilds_and_missing_points_recover(tmp_path):
    service, inner, client = make_service(tmp_path)
    first = service.ingest_text(text="Stable text.", source="paper.md")
    service.indexer.store.sparse_embeddings.model_id = "sparse-toy-v2"
    updated = service.ingest_text(text="Stable text.", source="paper.md")
    assert updated.document_version == 2 and updated.stats.sparse_cache_misses == 1
    assert updated.stats.vector_reuses == 0 and updated.stats.embedding_cache_hits == 1
    active = service.repository.active(first.document_id)
    client.delete(
        "evidence", points_selector=models.PointIdsList(points=active.chunk_ids), wait=True
    )
    recovered = service.ingest_text(text="Stable text.", source="paper.md")
    assert (
        recovered.document_version == 3
        and service.search("Stable")[0].evidence.document_version == 3
    )
    with pytest.raises(ValueError):
        service.indexer.activate(first.document_id, 2, "default")
    client.close()

"""Collection compatibility tests without downloading embedding models."""

from qdrant_client import QdrantClient

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.vectorstore import ensure_collection


def test_hybrid_collection_has_dense_and_bm25_vectors() -> None:
    settings = Settings(rag_collection_name="rag_test_collection")
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)
    ensure_collection(client, settings)
    config = client.get_collection(settings.rag_collection_name).config.params
    assert "dense" in config.vectors
    assert "sparse" in config.sparse_vectors

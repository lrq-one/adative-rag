"""Qdrant dense / BM25 sparse / fused retrieval with shared vector storage."""

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from ai_agent_template.core.settings import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    return FastEmbedEmbeddings(model_name=settings.rag_embedding_model_name)


def build_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        )
    return QdrantClient(path=settings.qdrant_storage_path)


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    name = settings.rag_collection_name
    if client.collection_exists(name):
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors or {}
        if not isinstance(vectors, dict) or "dense" not in vectors or "sparse" not in sparse_vectors:
            raise ValueError(
                f"Qdrant collection {name!r} uses an old dense-only schema. "
                "Choose a new AI_AGENT_RAG_COLLECTION_NAME to rebuild the hybrid index."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.rag_embedding_dimensions, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def build_retrieval_stores(settings: Settings) -> dict[str, QdrantVectorStore]:
    client = build_qdrant_client(settings)
    ensure_collection(client, settings)
    embedding = build_embeddings(settings)
    sparse = FastEmbedSparse(model_name=settings.rag_sparse_model_name)
    common = dict(
        client=client,
        collection_name=settings.rag_collection_name,
        embedding=embedding,
        sparse_embedding=sparse,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    return {
        "dense": QdrantVectorStore(**common, retrieval_mode=RetrievalMode.DENSE),
        "sparse": QdrantVectorStore(**common, retrieval_mode=RetrievalMode.SPARSE),
        "hybrid": QdrantVectorStore(**common, retrieval_mode=RetrievalMode.HYBRID),
    }


def build_vector_store(settings: Settings) -> QdrantVectorStore:
    """Backward-compatible builder; hybrid mode indexes both vector channels."""
    return build_retrieval_stores(settings)["hybrid"]

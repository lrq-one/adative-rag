"""Qdrant dense / BM25 sparse / fused retrieval with shared vector storage."""

from importlib.metadata import version
from pathlib import Path
from typing import Any

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import (
    CachedEmbeddings,
    CachedSparseEmbeddings,
    PersistentCache,
    model_artifacts,
    resolved_embedding_identity,
)
from ai_agent_template.rag.repository import fingerprint, timestamp
from ai_agent_template.rag.schemas import IndexManifest


def build_embeddings(settings: Settings) -> Embeddings:
    allowed_parameters = {
        "threads",
        "max_length",
        "doc_embed_type",
        "batch_size",
        "parallel",
        "providers",
        "cache_dir",
    }
    if set(settings.rag_embedding_parameters) - allowed_parameters:
        raise ValueError(
            "Unsupported FastEmbed encoding parameters; refusing silently ignored configuration"
        )
    if settings.rag_embedding_model_revision != "default":
        raise ValueError(
            "FastEmbed does not expose arbitrary revision selection. Use a registered pinned model ID instead."
        )
    inner = FastEmbedEmbeddings(
        model_name=settings.rag_embedding_model_name, **settings.rag_embedding_parameters
    )
    return CachedEmbeddings(
        inner,
        PersistentCache(Path(settings.rag_data_path) / "cache.sqlite3"),
        resolved_embedding_identity(settings, inner),
        settings.rag_embedding_dimensions,
    )


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
        if (
            not isinstance(vectors, dict)
            or "dense" not in vectors
            or "sparse" not in sparse_vectors
        ):
            raise ValueError(
                f"Qdrant collection {name!r} uses an old dense-only schema. "
                "Choose a new AI_AGENT_RAG_COLLECTION_NAME to rebuild the hybrid index."
            )
        if vectors["dense"].size != settings.rag_embedding_dimensions:
            raise ValueError("Collection vector dimensions do not match the embedding model")
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.rag_embedding_dimensions, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def build_retrieval_stores(settings: Settings) -> dict[str, QdrantVectorStore]:
    client = build_qdrant_client(settings)
    embedding = build_embeddings(settings)
    model_id = getattr(embedding, "model_id")
    sparse_inner = FastEmbedSparse(model_name=settings.rag_sparse_model_name)
    sparse_model = sparse_inner._model.model
    sparse_id = fingerprint(
        [
            settings.rag_sparse_model_name,
            version("fastembed"),
            {
                field: getattr(sparse_model, field, None)
                for field in (
                    "language",
                    "k",
                    "b",
                    "avg_len",
                    "token_max_length",
                    "disable_stemmer",
                )
            },
            sorted(getattr(sparse_model, "stopwords", [])),
            model_artifacts(sparse_model),
        ]
    )
    name = (
        settings.rag_collection_name + "_" + fingerprint([model_id, sparse_id, "evidence-v2"])[:12]
    )
    ensure_collection(client, settings.model_copy(update={"rag_collection_name": name}))
    cache = PersistentCache(Path(settings.rag_data_path) / "cache.sqlite3")
    sparse = CachedSparseEmbeddings(sparse_inner, cache, sparse_id)
    index = IndexManifest(
        collection_name=name,
        embedding_model_id=model_id,
        dimensions=settings.rag_embedding_dimensions,
        sparse_model_id=sparse_id,
        created_at=timestamp(),
    )
    cache.put_many("index_manifest", {name: index.model_dump()})
    common: dict[str, Any] = dict(
        client=client,
        collection_name=name,
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

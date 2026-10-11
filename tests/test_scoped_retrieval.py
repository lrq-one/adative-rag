from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.service import RagService
from ai_agent_template.rag.vectorstore import ensure_collection


class ToyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if "selected" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_source_and_type_filters_run_before_qdrant_top_k() -> None:
    settings = Settings(
        rag_collection_name="scope_test",
        rag_embedding_dimensions=2,
        rag_top_k=1,
        rag_search_multiplier=1,
    )
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)
    store = QdrantVectorStore(
        client=client,
        collection_name="scope_test",
        embedding=ToyEmbeddings(),
        vector_name="dense",
        retrieval_mode=RetrievalMode.DENSE,
    )
    store.add_documents(
        [
            Document(
                page_content="unselected high similarity",
                metadata={"source": "outside.pdf", "document_type": "knowledge"},
            ),
            Document(
                page_content="voice high similarity",
                metadata={"source": "inside.pdf", "document_type": "voice_sample"},
            ),
            Document(
                page_content="selected lower similarity",
                metadata={"source": "inside.pdf", "document_type": "knowledge"},
            ),
        ]
    )
    service = RagService(settings, store, retrieval_stores={"dense": store})
    result = service.search("query", mode="dense", source_ids=["inside.pdf"])
    assert len(result) == 1 and result[0].source == "inside.pdf"
    assert result[0].content == "selected lower similarity"

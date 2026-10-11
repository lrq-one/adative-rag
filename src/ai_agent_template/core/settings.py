from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_AGENT_",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    service_name: str = "scientific-adaptive-rag"
    log_level: str = "INFO"
    cors_allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    model_provider: Literal["openai", "groq"] = "groq"
    model_name: str = "llama-3.1-8b-instant"
    model_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_AGENT_MODEL_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        ),
    )
    model_temperature: float = 0
    model_timeout_seconds: float = 30
    model_max_retries: int = 2
    model_max_tokens: int = Field(default=2048, ge=128, le=8192)
    query_model_name: str | None = None
    query_model_max_tokens: int = Field(default=1200, ge=128, le=4096)
    rag_decision_token_budget: int = Field(default=16000, ge=256, le=100000)
    rag_max_search_calls: int = Field(default=9, ge=1, le=30)
    rag_context_max_chars: int = Field(default=12000, ge=1000, le=32000)
    agent_history_turns: int = Field(default=4, ge=1, le=10)

    rag_enabled: bool = True
    rag_data_path: str = ".rag"
    rag_namespace: str = "default"
    rag_verification_level: Literal["off", "deterministic", "semantic"] = "semantic"
    rag_verifier_model_name: str | None = None
    rag_answer_repair_limit: int = Field(default=1, ge=0, le=1)
    rag_collection_name: str = "scientific_knowledge"
    rag_top_k: int = 4
    rag_voice_top_k: int = 4
    rag_score_threshold: float | None = None
    rag_chunk_size: int = 1_000
    rag_chunk_overlap: int = 150
    rag_search_multiplier: int = 3
    rag_retrieval_mode: Literal["adaptive", "hybrid", "dense", "sparse"] = "adaptive"
    rag_sparse_model_name: str = "Qdrant/bm25"
    rag_reranker_enabled: bool = False
    rag_reranker_model: str = "BAAI/bge-reranker-base"
    rag_evidence_retry_limit: int = Field(default=2, ge=0, le=5)
    rag_ocr_enabled: bool = False
    rag_vlm_enabled: bool = False
    rag_vlm_model: str = "gpt-4o-mini"
    rag_parent_chunk_size: int = 2000
    rag_chunk_strategy: Literal["recursive", "section", "parent_child"] = "parent_child"
    rag_embedding_provider: Literal["fastembed"] = "fastembed"
    rag_embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    rag_embedding_dimensions: int = 384
    rag_embedding_model_revision: str = "default"
    rag_embedding_parameters: dict[str, Any] = Field(default_factory=dict)
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_storage_path: str = ".qdrant"

    auth_api_keys: list[str] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    request_id_header: str = "x-request-id"
    request_max_body_bytes: int = 5 * 1024 * 1024

    agent_checkpoint_backend: Literal["memory", "none"] = "memory"

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

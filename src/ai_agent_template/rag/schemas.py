from typing import Any, Literal

from pydantic import BaseModel, Field


class RagSource(BaseModel):
    source: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evidence: "EvidenceRecord | None" = None


class EvidenceRecord(BaseModel):
    evidence_id: str
    document_id: str
    document_version: int
    chunk_id: str
    parent_chunk_id: str | None = None
    source_filename: str
    page_number: int | None = None
    section_title: str | None = None
    content_type: str = "text"
    text: str
    bbox: tuple[float, float, float, float] | None = None
    figure_id: str | None = None
    table_id: str | None = None
    text_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationRecord(BaseModel):
    citation_id: str
    evidence: EvidenceRecord
    snippet: str
    status: Literal["active", "historical", "unavailable", "legacy"] = "active"


class IndexStats(BaseModel):
    total_chunks: int = 0
    unchanged_chunks: int = 0
    new_chunks: int = 0
    modified_chunks: int = 0
    deleted_chunks: int = 0
    metadata_only_updates: int = 0
    vector_reuses: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    sparse_cache_hits: int = 0
    sparse_cache_misses: int = 0
    parse_cache_hits: int = 0
    chunk_cache_hits: int = 0
    ocr_cache_hits: int = 0
    vlm_cache_hits: int = 0
    page_text_cache_hits: int = 0
    parsing_time_ms: float = 0
    chunking_time_ms: float = 0
    embedding_time_ms: float = 0
    qdrant_update_time_ms: float = 0
    total_index_time_ms: float = 0
    index_version: int = 0
    update_status: str = "building"
    error_type: str | None = None


class DocumentManifest(BaseModel):
    document_id: str
    document_version: int
    namespace: str = "default"
    source_filename: str
    document_type: Literal["knowledge", "voice_sample"] = "knowledge"
    file_sha256: str
    input_fingerprint: str = ""
    normalized_content_hash: str
    parser_version: str
    ocr_model_version: str | None = None
    vlm_model_version: str | None = None
    chunker_config_hash: str
    embedding_model_id: str
    index_schema_version: str = "evidence-v2"
    index_fingerprint: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    index_status: Literal["building", "active", "historical", "failed", "deleted"] = "building"
    created_at: str
    activated_at: str | None = None
    original_file: str | None = None
    stats: IndexStats = Field(default_factory=IndexStats)


class IndexManifest(BaseModel):
    collection_name: str
    embedding_model_id: str
    dimensions: int
    sparse_model_id: str
    schema_version: str = "evidence-v2"
    created_at: str


class RagIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=512)
    document_type: Literal["knowledge", "voice_sample"] = "knowledge"
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    namespace: str = Field(default="default", min_length=1, max_length=128)


class RagIngestResponse(BaseModel):
    source: str
    chunks: int
    document_id: str | None = None
    document_version: int | None = None
    stats: IndexStats | None = None


class RagSearchResponse(BaseModel):
    query: str
    sources: list[RagSource]

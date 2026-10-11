from typing import Annotated

from pydantic import BaseModel, Field

from ai_agent_template.agent.decisions import QueryPlan
from ai_agent_template.rag.evidence import VerificationReport
from ai_agent_template.rag.schemas import CitationRecord, RagSource


class AgentInvokeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_000)
    thread_id: str | None = Field(default=None, max_length=256)
    # Source IDs are the filenames/source names used at ingestion. None retains a thread's scope.
    active_doc_ids: list[Annotated[str, Field(min_length=1, max_length=512)]] | None = Field(
        default=None,
        max_length=8,
    )


class AgentInvokeResponse(BaseModel):
    message: str
    thread_id: str | None = None
    sources: list[RagSource] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    retrieval_rounds: int = 0
    search_calls: int = 0
    decision_tokens: int = 0
    stop_reason: str = ""
    verification_reason: str = ""
    missing_facts: list[str] = Field(default_factory=list)
    answer: str = ""
    citations: list[CitationRecord] = Field(default_factory=list)
    verification: VerificationReport | None = None
    answer_repairs: int = 0


class HealthResponse(BaseModel):
    status: str

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ai_agent_template.agent.decisions import EvidenceGrade, QueryPlan
from ai_agent_template.rag.evidence import GroundedClaim, VerificationReport
from ai_agent_template.rag.schemas import CitationRecord, RagSource


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[RagSource]
    voice_samples: list[RagSource]
    queries: list[str]
    attempts: int
    verification_reason: str
    original_query: str
    standalone_query: str
    query_plan: QueryPlan | None
    retrieval_plan: QueryPlan | None
    active_doc_ids: list[str]
    evidence_grade: EvidenceGrade
    answer_sources: list[RagSource]
    attempted_queries: list[str]
    seen_evidence: list[str]
    search_calls: int
    new_evidence: int
    decision_tokens: int
    final_answer: str
    stop_reason: str
    claims: list[GroundedClaim]
    citations: list[CitationRecord]
    verification: VerificationReport
    answer_repairs: int
    verification_calls: int
    verification_tokens: int
    generation_tokens: int

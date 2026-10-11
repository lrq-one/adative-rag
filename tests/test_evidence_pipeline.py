from typing import cast

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from ai_agent_template.agent.decisions import Decision, EvidenceGrade, QueryPlan
from ai_agent_template.agent.graph import build_agent_graph, initial_state
from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.evidence import (
    ClaimAssessment,
    ClaimAssessments,
    GroundedClaim,
    LLMEvidenceVerifier,
    VerificationReport,
    verify_claims,
)
from ai_agent_template.rag.repository import digest
from ai_agent_template.rag.schemas import EvidenceRecord, RagSource
from ai_agent_template.rag.service import RagService
from tests.test_query_policy import StructuredModel
from tests.test_rag_service import FakeVectorStore


def record(text="ModelA MAE on SetA was 25.05 s.", doc="doc-a", version=1, chunk="a"):
    return EvidenceRecord(
        evidence_id=chunk,
        document_id=doc,
        document_version=version,
        chunk_id=chunk,
        source_filename="paper.pdf",
        page_number=3,
        section_title="Results",
        text=text,
        text_hash=digest(text),
    )


@pytest.mark.parametrize(
    "text,ids,expected",
    [
        ("ModelA MAE on SetA was 25.05 s.", ["E1"], True),
        ("ModelA MAE on SetA was 25.05 s.", ["E9"], False),
        ("ModelB MAE on SetA was 25.05 s.", ["E1"], False),
        ("ModelA MAE on SetB was 25.05 s.", ["E1"], False),
        ("ModelA MAE on SetA was 25.05 ms.", ["E1"], False),
        ("ModelA MAE on SetA was 25050 ms.", ["E1"], True),
    ],
)
async def test_claim_bound_deterministic_checks(text, ids, expected):
    result = await verify_claims(
        [GroundedClaim(text=text, evidence_ids=ids)],
        {"E1": record()},
        level="deterministic",
        entities=["ModelA", "ModelB", "SetA", "SetB"],
    )
    assert result.supported == expected


async def test_same_document_other_chunk_cannot_supply_number():
    result = await verify_claims(
        [GroundedClaim(text="MAE was 25.05 s.", evidence_ids=["E2"])],
        {"E1": record(), "E2": record("The method was evaluated.", chunk="b")},
        level="deterministic",
    )
    assert not result.supported


async def test_stale_version_is_rejected():
    result = await verify_claims(
        [GroundedClaim(text="MAE was 25.05 s.", evidence_ids=["E1"])],
        {"E1": record()},
        level="deterministic",
        status=lambda _: "historical",
    )
    assert not result.supported


async def test_missing_semantic_judge_is_not_a_pass():
    result = await verify_claims(
        [GroundedClaim(text="MAE was 25.05 s.", evidence_ids=["E1"])],
        {"E1": record()},
        level="semantic",
    )
    assert not result.supported and result.error == "semantic_judge_not_configured"


def test_positions_and_same_filenames_keep_distinct_identity(tmp_path):
    service = RagService(Settings(rag_data_path=str(tmp_path)), FakeVectorStore())
    first = service.ingest_sections(
        sections=[
            Document(page_content="Repeated.", metadata={"page": 1}),
            Document(page_content="Repeated.", metadata={"page": 2}),
        ],
        source="paper.pdf",
        document_id="first",
    )
    second = service.ingest_text(text="Repeated.", source="paper.pdf", document_id="second")
    one = service.repository.versions(first.document_id)[0]
    assert len(one.chunk_ids) == len(set(one.chunk_ids)) == 2
    assert first.document_id != second.document_id
    assert service.repository.resolve(one.chunk_ids[1])[0].page_number == 2


def test_old_evidence_resolves_historical_and_delete_hides_all_versions(tmp_path):
    service = RagService(Settings(rag_data_path=str(tmp_path)), FakeVectorStore())
    first = service.ingest_text(text="Old.", source="paper.pdf")
    old_id = service.repository.versions(first.document_id)[0].chunk_ids[0]
    service.ingest_text(text="New.", source="paper.pdf")
    assert service.repository.resolve(old_id)[1] == "historical"
    assert all(source.content == "New." for source in service.search("text"))
    service.delete_source(first.document_id)
    assert service.repository.resolve(old_id) is None and service.search("text") == []


class Policy:
    async def plan(self, question, *args, **kwargs):
        return Decision(QueryPlan(standalone_query=question, intent="fact_lookup", route="dense"))

    async def grade(self, *args, **kwargs):
        return Decision(EvidenceGrade(sufficient=True, relevant_indices=[0], covered_facts=["MAE"]))


class Retriever:
    def search(self, *args, **kwargs):
        return [RagSource(source="paper.pdf", content=record().text, evidence=record())]


class Writer:
    def __init__(self, answers):
        self.answers = iter(answers)

    async def ainvoke(self, *args, **kwargs):
        return AIMessage(content=next(self.answers))


async def test_one_answer_repair_then_verified_citations():
    graph = build_agent_graph(
        Writer(["MAE was 30 s. [E9]", "MAE was 25.05 s. [E1]"]), Retriever(), query_policy=Policy()
    )
    result = await graph.ainvoke(initial_state("MAE?"))
    assert result["answer_repairs"] == 1
    assert result["citations"][0].evidence.page_number == 3
    assert result["final_answer"] == "MAE was 25.05 s. [E1]"


async def test_failed_repair_never_streams_draft():
    graph = build_agent_graph(
        Writer(["MAE was 30 s. [E9]"] * 2), Retriever(), query_policy=Policy()
    )
    result = await graph.ainvoke(initial_state("MAE?"))
    assert result["stop_reason"] == "verification_failed" and "30 s" not in result["final_answer"]


class Judge:
    async def assess(self, claims, records):
        checks = [
            ClaimAssessment(
                claim_index=i,
                claim=claim.text,
                evidence_ids=claim.evidence_ids,
                verdict="supported" if i == 0 else "uncertain",
                reason="conflicting evidence" if i else "",
                failure_type="conflict" if i else "none",
                conflicting_evidence_ids=["E1"] if i else [],
            )
            for i, claim in enumerate(claims)
        ]
        return VerificationReport(level="semantic", supported=False, checks=checks)


async def test_partial_semantic_support_filters_other_facts():
    graph = build_agent_graph(
        Writer(["MAE was 25.05 s. [E1]\nThis model is best. [E1]"]),
        Retriever(),
        query_policy=Policy(),
        verification_level="semantic",
        evidence_verifier=Judge(),
        answer_repair_limit=0,
    )
    result = await graph.ainvoke(initial_state("Model result?"))
    assert "25.05" in result["final_answer"] and "best" not in result["final_answer"]
    assert result["stop_reason"] == "partial_verified"


@pytest.mark.parametrize(
    "bad", ["incomplete", "changed_claim", "unknown_conflict", "provider_error"]
)
async def test_semantic_judge_malformed_or_unavailable_fails_closed(bad):
    claims = [GroundedClaim(text="MAE was 25.05 s.", evidence_ids=["E1"])]
    check = ClaimAssessment(
        claim_index=0,
        claim=claims[0].text,
        evidence_ids=["E1"],
        verdict="supported",
        reason="supported",
    )
    if bad == "changed_claim":
        check.claim = "A different statement"
    if bad == "unknown_conflict":
        check.conflicting_evidence_ids = ["E99"]
    model = StructuredModel(
        ClaimAssessments(checks=[] if bad == "incomplete" else [check]),
        error=ValueError("unavailable") if bad == "provider_error" else None,
    )
    result = await LLMEvidenceVerifier(cast(BaseChatModel, model)).assess(claims, {"E1": record()})
    assert not result.supported and result.error


async def test_semantic_conflicting_evidence_is_not_accepted():
    claim = GroundedClaim(text="ModelA MAE was 25.05 s.", evidence_ids=["E1", "E2"])
    check = ClaimAssessment(
        claim_index=0,
        claim=claim.text,
        evidence_ids=claim.evidence_ids,
        verdict="unsupported",
        failure_type="conflict",
        reason="Conflicting conditions",
        conflicting_evidence_ids=["E2"],
    )
    model = StructuredModel(ClaimAssessments(checks=[check]))
    result = await verify_claims(
        [claim],
        {"E1": record(), "E2": record("ModelA MAE was 26.05 s.", chunk="b")},
        level="semantic",
        judge=LLMEvidenceVerifier(cast(BaseChatModel, model)),
    )
    assert not result.supported and result.checks[0].failure_type == "conflict"

"""Unit coverage for adaptive routing, evidence gating and offline retrieval metrics."""

import pytest
from langchain_core.messages import AIMessage

from ai_agent_template.agent.graph import build_agent_graph, initial_state
from ai_agent_template.evaluation.metrics import evaluate_retrieval
from ai_agent_template.rag.schemas import RagSource
from ai_agent_template.rag.service import choose_retrieval_mode
from ai_agent_template.rag.verification import validate_citations


def test_adaptive_query_routing() -> None:
    assert choose_retrieval_mode("What is CWN in SMRT?") == "sparse"
    assert choose_retrieval_mode("为什么保留时间预测需要严格划分数据") == "dense"
    assert choose_retrieval_mode("比较两个模型在同一数据集上的表现") == "hybrid"


def test_citation_verifier_rejects_untraceable_claims() -> None:
    sources = [RagSource(source="paper.pdf", content="The MAE was 25.05 seconds")]
    assert validate_citations("The MAE was 25.05 [paper.pdf]", sources).supported
    assert not validate_citations("The MAE was 12.3 [paper.pdf]", sources).supported
    assert not validate_citations("The MAE was 25.05 [invented.pdf]", sources).supported
    assert not validate_citations("The MAE was 25.05", sources).supported


def test_retrieval_metrics() -> None:
    result = evaluate_retrieval([
        {"gold_sources": ["a", "b"], "retrieved_sources": ["x", "a", "c"]},
        {"gold_sources": ["d"], "retrieved_sources": ["d"]},
    ], k=3)
    assert result.evaluated == 2
    assert result.recall_at_k == pytest.approx(0.75)
    assert result.hit_rate_at_k == pytest.approx(1.0)
    assert result.mrr_at_k == pytest.approx(0.75)


class StaticModel:
    async def ainvoke(self, messages):
        return AIMessage(content="The MAE was 25.05 [paper.pdf]")


class StaticRetriever:
    def search(self, query):
        return [RagSource(source="paper.pdf", content="The MAE was 25.05 seconds")]

    def search_voice(self, query):
        return []


@pytest.mark.asyncio
async def test_workflow_generates_checked_answer() -> None:
    graph = build_agent_graph(StaticModel(), retriever=StaticRetriever())
    result = await graph.ainvoke(initial_state("What was the MAE?"))
    assert str(result["messages"][-1].content) == "The MAE was 25.05 [paper.pdf]"


class EmptyRetriever:
    def search(self, query):
        return []

    def search_voice(self, query):
        return []


@pytest.mark.asyncio
async def test_workflow_refuses_when_evidence_missing() -> None:
    graph = build_agent_graph(StaticModel(), retriever=EmptyRetriever(), max_retries=1)
    result = await graph.ainvoke(initial_state("Unknown model?"))
    assert "没有检索到足够证据" in str(result["messages"][-1].content)

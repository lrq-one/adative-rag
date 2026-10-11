"""Provider-output validation and budget accounting without network access."""

import json
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from ai_agent_template.agent.decisions import EvidenceGrade, LLMQueryPolicy, QueryPlan
from ai_agent_template.rag.schemas import RagSource


class StructuredModel:
    def __init__(self, parsed: Any, error: Any = None) -> None:
        self.parsed, self.error = parsed, error
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "StructuredModel":
        assert kwargs == {"method": "function_calling", "include_raw": True}
        self.schema = schema
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"data": json.loads(messages[-1].content), **kwargs})
        return {
            "parsed": self.parsed,
            "parsing_error": self.error,
            "raw": AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            ),
        }


@pytest.mark.asyncio
async def test_structured_plan_preserves_context_and_counts_provider_usage() -> None:
    value = QueryPlan(
        standalone_query="TC-TopoRT MAE on SMRT", intent="fact_lookup", route="sparse"
    )
    model = StructuredModel(value)
    policy = LLMQueryPolicy(cast(BaseChatModel, model))
    result = await policy.plan(
        "它的 MAE 呢？",
        [AIMessage(content="TC-TopoRT on SMRT")],
        ["paper.pdf"],
        [],
        remaining_tokens=8000,
    )
    assert result.value == value and result.tokens == 120
    assert model.calls[0]["data"]["original_query"] == "它的 MAE 呢？"
    assert model.calls[0]["data"]["active_doc_ids"] == ["paper.pdf"]
    assert model.calls[0]["max_tokens"] <= 1200


@pytest.mark.asyncio
async def test_invalid_structured_output_is_not_a_rule_based_plan() -> None:
    model = StructuredModel(None, error=ValueError("malformed JSON"))
    result = await LLMQueryPolicy(cast(BaseChatModel, model)).plan(
        "question", [], [], [], remaining_tokens=8000
    )
    assert result.value is None and result.error == "invalid_structured_output"
    assert result.tokens == 120


@pytest.mark.asyncio
async def test_too_small_token_budget_never_calls_provider() -> None:
    model = StructuredModel(None)
    result = await LLMQueryPolicy(cast(BaseChatModel, model)).plan(
        "question", [], [], [], remaining_tokens=10
    )
    assert result.error == "decision_token_budget" and not model.calls


@pytest.mark.asyncio
async def test_grader_cannot_reference_evidence_outside_prompt_window() -> None:
    model = StructuredModel(EvidenceGrade(sufficient=True, relevant_indices=[1]))
    policy = LLMQueryPolicy(cast(BaseChatModel, model), context_max_chars=100)
    sources = [RagSource(source="a", content="x" * 200), RagSource(source="b", content="answer")]
    plan = QueryPlan(standalone_query="question", intent="fact_lookup", route="sparse")
    result = await policy.grade(plan, sources, remaining_tokens=8000)
    assert len(model.calls[0]["data"]["evidence"]) == 1
    assert result.value is None and result.error == "invalid_evidence_reference"


def test_ambiguous_plan_requires_question_and_bounds_subqueries() -> None:
    with pytest.raises(ValueError, match="clarification question"):
        QueryPlan(
            standalone_query="it", intent="fact_lookup", route="dense", needs_clarification=True
        )
    with pytest.raises(ValueError):
        QueryPlan(standalone_query="q", intent="multi_hop", route="hybrid", sub_queries=["q"] * 4)


@pytest.mark.asyncio
async def test_unbacked_abbreviation_expansion_is_removed_from_plan() -> None:
    value = QueryPlan(
        standalone_query="为什么RT（反应时）预测必须核对数据划分？",
        intent="explanation",
        route="dense",
        sub_queries=["RT（回归时间）数据划分"],
        required_facts=["RT（反应时）的比较条件"],
    )
    result = await LLMQueryPolicy(cast(BaseChatModel, StructuredModel(value))).plan(
        "为什么RT预测必须核对数据划分？", [], [], [], remaining_tokens=8000
    )
    assert result.value and result.value.standalone_query == "为什么RT预测必须核对数据划分？"
    assert result.value.sub_queries == ["RT数据划分"]
    assert result.value.required_facts == ["RT的比较条件"]


@pytest.mark.asyncio
async def test_abbreviation_expansion_can_use_explicit_previous_evidence() -> None:
    value = QueryPlan(
        standalone_query="RT（保留时间）预测的比较条件", intent="fact_lookup", route="dense"
    )
    result = await LLMQueryPolicy(cast(BaseChatModel, StructuredModel(value))).plan(
        "RT比较条件呢？",
        [],
        [],
        [RagSource(source="paper", content="保留时间 RT 的比较条件")],
        remaining_tokens=8000,
    )
    assert result.value and "保留时间" in result.value.standalone_query

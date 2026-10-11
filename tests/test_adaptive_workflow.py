"""Behavioral tests for semantic routing, evidence feedback and conversation boundaries."""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_template.agent.decisions import Decision, EvidenceGrade, QueryPlan
from ai_agent_template.agent.graph import build_agent_graph, initial_state
from ai_agent_template.agent.service import AgentService
from ai_agent_template.api.schemas import AgentInvokeRequest
from ai_agent_template.rag.schemas import RagSource

A = RagSource(source="a.pdf", content="TC-TopoRT MAE was 25.05 seconds on SMRT.")
B = RagSource(source="b.pdf", content="GNN MAE was 30.00 seconds on the same SMRT test split.")


def make_plan(query: str = "Compare TC-TopoRT and GNN MAE on SMRT", **kwargs: Any) -> QueryPlan:
    return QueryPlan(
        standalone_query=query,
        intent="comparison",
        route="hybrid",
        required_facts=["TC-TopoRT MAE", "GNN MAE"],
        **kwargs,
    )


class ScriptedPolicy:
    def __init__(
        self,
        plans: list[QueryPlan],
        grades: list[EvidenceGrade] | None = None,
        rewrites: list[QueryPlan] | None = None,
        tokens: int = 10,
    ) -> None:
        self.plans = iter(plans)
        self.grades = iter(grades or [])
        self.rewrites = iter(rewrites or [])
        self.tokens = tokens
        self.histories: list[list[BaseMessage]] = []
        self.previous_sources: list[list[RagSource]] = []
        self.feedback: list[tuple[list[str], list[str]]] = []
        self.graded_plans: list[QueryPlan] = []

    async def plan(
        self,
        question: str,
        history: Sequence[BaseMessage],
        active_doc_ids: Sequence[str],
        previous_sources: Sequence[RagSource],
        *,
        remaining_tokens: int,
    ) -> Decision[QueryPlan]:
        self.histories.append(list(history))
        self.previous_sources.append(list(previous_sources))
        return Decision(next(self.plans), tokens=self.tokens)

    async def grade(
        self, plan: QueryPlan, sources: Sequence[RagSource], *, remaining_tokens: int
    ) -> Decision[EvidenceGrade]:
        self.graded_plans.append(plan)
        return Decision(next(self.grades), tokens=self.tokens)

    async def rewrite(
        self,
        plan: QueryPlan,
        sources: Sequence[RagSource],
        grade: EvidenceGrade,
        attempted_queries: Sequence[str],
        *,
        remaining_tokens: int,
    ) -> Decision[QueryPlan]:
        self.feedback.append((list(grade.missing_facts), list(attempted_queries)))
        return Decision(next(self.rewrites), tokens=self.tokens)


class RecordingRetriever:
    def __init__(self, batches: list[list[RagSource]]) -> None:
        self.batches = iter(batches)
        self.calls: list[tuple[str, str | None, list[str]]] = []

    def search(
        self, query: str, *, mode: str | None = None, source_ids: Sequence[str] | None = None
    ) -> list[RagSource]:
        self.calls.append((query, mode, list(source_ids or [])))
        return next(self.batches)

    def search_voice(self, query: str) -> list[RagSource]:
        return []


class AnswerModel:
    def __init__(self, answer: str = "The MAE was 25.05 [a.pdf]") -> None:
        self.answer = answer
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.answer)


@pytest.mark.asyncio
async def test_llm_route_overrides_keyword_route() -> None:
    plan = QueryPlan(standalone_query="Explain MAE on SMRT", intent="explanation", route="dense")
    policy = ScriptedPolicy([plan], [EvidenceGrade(sufficient=True, relevant_indices=[0])])
    retriever = RecordingRetriever([[A]])
    result = await build_agent_graph(AnswerModel(), retriever, query_policy=policy).ainvoke(
        initial_state("What is MAE in SMRT?")
    )
    assert retriever.calls == [(plan.standalone_query, "dense", [])]
    assert result["stop_reason"] == "answered"


@pytest.mark.asyncio
async def test_answer_prompt_supplies_exact_allowed_citations() -> None:
    policy = ScriptedPolicy([make_plan()], [EvidenceGrade(sufficient=True, relevant_indices=[0])])
    model = AnswerModel()
    await build_agent_graph(model, RecordingRetriever([[A]]), query_policy=policy).ainvoke(
        initial_state("What was the MAE?")
    )
    prompt = str(model.calls[0][0].content)
    assert 'Allowed citations: ["[a.pdf]"]' in prompt
    assert "[source]" not in prompt


@pytest.mark.asyncio
async def test_ambiguous_query_clarifies_without_retrieval() -> None:
    plan = make_plan(
        "它的效果呢？", needs_clarification=True, clarification_question="你指的是哪个模型？"
    )
    retriever, model = RecordingRetriever([]), AnswerModel()
    result = await build_agent_graph(model, retriever, query_policy=ScriptedPolicy([plan])).ainvoke(
        initial_state("它的效果呢？")
    )
    assert result["final_answer"] == "你指的是哪个模型？"
    assert result["attempts"] == 0
    assert not retriever.calls and not model.calls


@pytest.mark.asyncio
async def test_nonempty_irrelevant_evidence_does_not_generate_answer() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                missing_facts=["GNN MAE"],
                relevant_indices=[],
            )
        ],
    )
    model = AnswerModel()
    result = await build_agent_graph(
        model, RecordingRetriever([[A]]), max_retries=0, query_policy=policy
    ).ainvoke(initial_state("Compare models"))
    assert not model.calls
    assert "没有检索到足够证据" in result["final_answer"]


@pytest.mark.asyncio
async def test_missing_evidence_is_not_a_preference_clarification() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                relevant_indices=[0],
                covered_facts=["TC-TopoRT MAE"],
                missing_facts=["GNN MAE"],
                clarification_question="Would you like more detail?",
                clarification_reason="ambiguous_document",
            )
        ],
        [make_plan(sub_queries=["GNN MAE on SMRT"])],
    )
    result = await build_agent_graph(
        AnswerModel(), RecordingRetriever([[A], [A]]), query_policy=policy
    ).ainvoke(initial_state("Compare models"))
    assert result["attempts"] == 2 and result["stop_reason"] == "no_new_evidence"
    assert "部分回答" in result["final_answer"] and "Would you like" not in result["final_answer"]


@pytest.mark.asyncio
async def test_rewrite_cannot_turn_missing_evidence_into_user_question() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                relevant_indices=[0],
                covered_facts=["TC-TopoRT MAE"],
                missing_facts=["GNN MAE"],
            )
        ],
        [
            make_plan(
                needs_clarification=True,
                clarification_question="Would you like help finding papers?",
            )
        ],
    )
    retriever = RecordingRetriever([[A]])
    result = await build_agent_graph(AnswerModel(), retriever, query_policy=policy).ainvoke(
        initial_state("Compare models")
    )
    assert len(retriever.calls) == 1 and result["stop_reason"] == "no_useful_rewrite"
    assert "部分回答" in result["final_answer"] and "Would you like" not in result["final_answer"]


@pytest.mark.asyncio
async def test_real_comparison_ambiguity_can_request_user_input() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                relevant_indices=[0, 1],
                missing_facts=["which test split"],
                conflicts=["Two papers report results on different test splits"],
                clarification_question="Which test split should be compared?",
                clarification_reason="unspecified_comparison_conditions",
                clarification_evidence_indices=[0, 1],
            )
        ],
    )
    result = await build_agent_graph(
        AnswerModel(), RecordingRetriever([[A, B]]), query_policy=policy
    ).ainvoke(initial_state("Compare models"))
    assert result["stop_reason"] == "clarification_required" and result["attempts"] == 1


@pytest.mark.asyncio
async def test_missing_fact_drives_targeted_retry_and_accumulates_evidence() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                relevant_indices=[0],
                covered_facts=["TC-TopoRT MAE"],
                missing_facts=["GNN MAE"],
            ),
            EvidenceGrade(
                sufficient=True, relevant_indices=[0, 1], covered_facts=["TC-TopoRT MAE", "GNN MAE"]
            ),
        ],
        [make_plan(sub_queries=["GNN MAE on the SMRT test split"])],
    )
    retriever = RecordingRetriever([[A], [B]])
    result = await build_agent_graph(AnswerModel(), retriever, query_policy=policy).ainvoke(
        initial_state("Compare models")
    )
    assert result["attempts"] == 2
    assert {source.source for source in result["answer_sources"]} == {"a.pdf", "b.pdf"}
    assert retriever.calls[1][0] == "GNN MAE on the SMRT test split"
    assert policy.feedback == [(["GNN MAE"], [make_plan().standalone_query])]
    assert policy.graded_plans[0] == policy.graded_plans[1]  # Never weaken original requirements.


@pytest.mark.asyncio
async def test_duplicate_query_stops_without_executing_again() -> None:
    first = make_plan("Find GNN MAE")
    policy = ScriptedPolicy(
        [first],
        [EvidenceGrade(sufficient=False, missing_facts=["GNN MAE"])],
        [make_plan(sub_queries=["find gnn mae!!!"])],
    )
    retriever = RecordingRetriever([[A]])
    result = await build_agent_graph(AnswerModel(), retriever, query_policy=policy).ainvoke(
        initial_state("Find GNN MAE")
    )
    assert len(retriever.calls) == 1
    assert result["stop_reason"] == "repeated_query"


@pytest.mark.asyncio
async def test_no_gain_stops_early_with_partial_evidence() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=False,
                relevant_indices=[0],
                covered_facts=["TC-TopoRT MAE"],
                missing_facts=["GNN MAE"],
            )
        ],
        [make_plan(sub_queries=["GNN MAE results"])],
    )
    model = AnswerModel()
    result = await build_agent_graph(
        model, RecordingRetriever([[A], [A]]), query_policy=policy
    ).ainvoke(initial_state("Compare models"))
    assert result["stop_reason"] == "no_new_evidence"
    assert len(policy.graded_plans) == 1
    assert "GNN MAE" in str(model.calls[0][-1].content)
    assert result["answer_sources"] == [A]


@pytest.mark.asyncio
async def test_round_budget_is_initial_round_plus_two_retries() -> None:
    grades = [EvidenceGrade(sufficient=False, missing_facts=["GNN MAE"]) for _ in range(3)]
    policy = ScriptedPolicy(
        [make_plan()],
        grades,
        [
            make_plan(sub_queries=["GNN test split MAE"]),
            make_plan(sub_queries=["GNN benchmark evaluation table"]),
        ],
    )
    extra = RagSource(source="c.pdf", content="Unrelated benchmark notes")
    retriever = RecordingRetriever([[A], [B], [extra]])
    result = await build_agent_graph(AnswerModel(), retriever, query_policy=policy).ainvoke(
        initial_state("Compare models")
    )
    assert result["attempts"] == 3 and len(retriever.calls) == 3
    assert result["stop_reason"] == "retrieval_round_budget"


@pytest.mark.asyncio
async def test_search_call_budget_caps_decomposed_queries() -> None:
    plan = make_plan(sub_queries=["one", "two", "three"])
    policy = ScriptedPolicy([plan], [EvidenceGrade(sufficient=False, missing_facts=["GNN MAE"])])
    retriever = RecordingRetriever([[A]])
    result = await build_agent_graph(
        AnswerModel(), retriever, query_policy=policy, max_search_calls=1
    ).ainvoke(initial_state("Compare models"))
    assert len(retriever.calls) == 1 and result["search_calls"] == 1
    assert result["stop_reason"] == "search_budget"


@pytest.mark.asyncio
async def test_exhausted_decision_budget_makes_no_more_model_or_search_calls() -> None:
    policy = ScriptedPolicy([make_plan()], tokens=100)
    retriever = RecordingRetriever([])
    result = await build_agent_graph(
        AnswerModel(), retriever, query_policy=policy, decision_token_budget=100
    ).ainvoke(initial_state("Compare models"))
    assert not retriever.calls and not policy.graded_plans
    assert result["stop_reason"] == "decision_token_budget"


@pytest.mark.asyncio
async def test_bad_evidence_indices_cannot_bypass_gate() -> None:
    policy = ScriptedPolicy([make_plan()], [EvidenceGrade(sufficient=True, relevant_indices=[999])])
    model = AnswerModel()
    result = await build_agent_graph(model, RecordingRetriever([[A]]), query_policy=policy).ainvoke(
        initial_state("Compare models")
    )
    assert not model.calls and "没有检索到足够证据" in result["final_answer"]


@pytest.mark.asyncio
async def test_condition_mismatch_prevents_sufficient_grade() -> None:
    policy = ScriptedPolicy(
        [make_plan()],
        [
            EvidenceGrade(
                sufficient=True,
                relevant_indices=[0],
                condition_mismatches=["different test split"],
            )
        ],
    )
    model = AnswerModel()
    result = await build_agent_graph(
        model, RecordingRetriever([[A]]), query_policy=policy, max_retries=0
    ).ainvoke(initial_state("Compare models"))
    assert not model.calls and "different test split" in result["final_answer"]


@pytest.mark.asyncio
async def test_thread_history_and_document_scope_are_preserved_and_isolated() -> None:
    policy = ScriptedPolicy(
        [make_plan("First question"), make_plan("TC-TopoRT MAE"), make_plan("New conversation")],
        [EvidenceGrade(sufficient=True, relevant_indices=[0]) for _ in range(3)],
    )
    retriever = RecordingRetriever([[A], [A], [A]])
    graph = build_agent_graph(
        AnswerModel(), retriever, query_policy=policy, checkpointer=InMemorySaver()
    )
    first_config = {"configurable": {"thread_id": "one"}}
    await graph.ainvoke(initial_state("First question", ["a.pdf"]), config=first_config)
    await graph.ainvoke(initial_state("它的 MAE 呢？"), config=first_config)
    await graph.ainvoke(
        initial_state("New conversation"), config={"configurable": {"thread_id": "two"}}
    )
    assert isinstance(policy.histories[1][0], HumanMessage)
    assert policy.histories[1][0].content == "First question"
    assert policy.previous_sources[1] == [A]
    assert not policy.histories[2] and not policy.previous_sources[2]
    assert retriever.calls[1] == ("TC-TopoRT MAE", "hybrid", ["a.pdf"])
    assert retriever.calls[2][2] == []


@pytest.mark.asyncio
async def test_document_scope_is_enforced_even_if_retriever_returns_other_sources() -> None:
    policy = ScriptedPolicy([make_plan()], [EvidenceGrade(sufficient=True, relevant_indices=[0])])
    result = await build_agent_graph(
        AnswerModel(), RecordingRetriever([[B, A]]), query_policy=policy
    ).ainvoke(initial_state("Find MAE", ["a.pdf"]))
    assert result["answer_sources"] == [A]


@pytest.mark.asyncio
async def test_stream_never_leaks_rejected_draft_or_decision_output() -> None:
    policy = ScriptedPolicy([make_plan()], [EvidenceGrade(sufficient=True, relevant_indices=[0])])
    graph = build_agent_graph(
        AnswerModel("Invented MAE 999 [a.pdf]"), RecordingRetriever([[A]]), query_policy=policy
    )
    events = [
        event async for event in AgentService(graph).stream(AgentInvokeRequest(message="Find MAE"))
    ]
    text = "".join(event["text"] for event in events if event["type"] == "delta")
    assert "999" not in text and "不足以验证" in text
    assert events[-1]["stop_reason"] == "verification_failed"


@pytest.mark.asyncio
async def test_technical_failure_is_not_a_retrieval_retry() -> None:
    class UnavailablePolicy(ScriptedPolicy):
        async def plan(self, *args: Any, **kwargs: Any) -> Decision[QueryPlan]:
            raise TimeoutError("provider unavailable")

    retriever = RecordingRetriever([])
    graph = build_agent_graph(AnswerModel(), retriever, query_policy=UnavailablePolicy([]))
    with pytest.raises(TimeoutError, match="provider unavailable"):
        await graph.ainvoke(initial_state("Find MAE"))
    assert not retriever.calls


@pytest.mark.asyncio
async def test_failed_new_plan_does_not_reuse_previous_turn_plan_or_sources() -> None:
    class FlakyPolicy(ScriptedPolicy):
        async def plan(self, *args: Any, **kwargs: Any) -> Decision[QueryPlan]:
            if self.histories:
                return Decision(None, tokens=10, error="invalid_structured_output")
            return await super().plan(*args, **kwargs)

    policy = FlakyPolicy([make_plan()], [EvidenceGrade(sufficient=True, relevant_indices=[0])])
    graph = build_agent_graph(
        AnswerModel(), RecordingRetriever([[A]]), query_policy=policy, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "same"}}
    await graph.ainvoke(initial_state("First question"), config=config)
    result = await graph.ainvoke(initial_state("New question"), config=config)
    assert result["query_plan"] is None and not result["sources"] and not result["answer_sources"]
    assert result["stop_reason"] == "invalid_structured_output"


@pytest.mark.asyncio
async def test_history_window_is_bounded_in_checkpoint_and_planning_context() -> None:
    policy = ScriptedPolicy(
        [make_plan(f"question {index}") for index in range(5)],
        [EvidenceGrade(sufficient=True, relevant_indices=[0]) for _ in range(5)],
    )
    graph = build_agent_graph(
        AnswerModel(),
        RecordingRetriever([[A]] * 5),
        query_policy=policy,
        checkpointer=InMemorySaver(),
        history_turns=1,
    )
    config = {"configurable": {"thread_id": "bounded"}}
    for index in range(5):
        await graph.ainvoke(initial_state(f"question {index}"), config=config)
    assert all(len(history) <= 2 for history in policy.histories)
    assert policy.histories[-1][0].content == "question 3"
    assert len(graph.get_state(config).values["messages"]) <= 4


@pytest.mark.asyncio
async def test_anonymous_requests_do_not_share_checkpoint_history() -> None:
    policy = ScriptedPolicy(
        [make_plan("one"), make_plan("two")],
        [EvidenceGrade(sufficient=True, relevant_indices=[0]) for _ in range(2)],
    )
    graph = build_agent_graph(
        AnswerModel(),
        RecordingRetriever([[A], [A]]),
        query_policy=policy,
        checkpointer=InMemorySaver(),
    )
    service = AgentService(graph)
    await service.invoke(AgentInvokeRequest(message="one"))
    await service.invoke(AgentInvokeRequest(message="two"))
    assert policy.histories == [[], []] and policy.previous_sources == [[], []]

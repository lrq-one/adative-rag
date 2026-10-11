from asyncio import Lock
from collections.abc import AsyncIterator
from typing import Any, cast
from weakref import WeakValueDictionary

from langchain_core.messages import AIMessage

from ai_agent_template.agent.decisions import EvidenceGrade, QueryPlan
from ai_agent_template.agent.graph import initial_state
from ai_agent_template.api.schemas import AgentInvokeRequest, AgentInvokeResponse
from ai_agent_template.observability.tracing import agent_span
from ai_agent_template.rag.schemas import RagSource

AgentGraph = Any


def _details(result: dict[str, Any]) -> dict[str, Any]:
    grade = result.get("evidence_grade")
    missing = (
        [*grade.missing_facts, *grade.condition_mismatches, *grade.conflicts]
        if isinstance(grade, EvidenceGrade)
        else []
    )
    plan = result.get("query_plan")
    return {
        "query_plan": plan.model_dump() if isinstance(plan, QueryPlan) else None,
        "retrieval_rounds": result.get("attempts", 0),
        "search_calls": result.get("search_calls", 0),
        "decision_tokens": result.get("decision_tokens", 0),
        "stop_reason": result.get("stop_reason", ""),
        "verification_reason": result.get("verification_reason", ""),
        "missing_facts": list(dict.fromkeys(missing)),
        "answer": result.get("final_answer", ""),
        "citations": [citation.model_dump() for citation in result.get("citations", [])],
        "verification": result["verification"].model_dump() if result.get("verification") else None,
        "answer_repairs": result.get("answer_repairs", 0),
    }


class AgentService:
    def __init__(self, graph: AgentGraph) -> None:
        self._graph = graph
        self._thread_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()

    def _thread_lock(self, config: dict[str, Any]) -> Lock:
        thread_id = str(config["configurable"]["thread_id"])
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = Lock()
            self._thread_locks[thread_id] = lock
        return lock

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponse:
        config = self._config(request.thread_id)
        async with self._thread_lock(config):
            with agent_span("agent.invoke", thread_id=request.thread_id):
                result = cast(
                    dict[str, Any],
                    await self._graph.ainvoke(
                        initial_state(request.message, request.active_doc_ids),
                        config=config,
                    ),
                )
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else AIMessage(content="")
        sources = [
            source
            for source in result.get("answer_sources", result.get("sources", []))
            if isinstance(source, RagSource)
        ]
        return AgentInvokeResponse(
            message=result.get("final_answer", str(final_message.content)),
            thread_id=request.thread_id,
            sources=sources,
            **_details(result),
        )

    async def stream(self, request: AgentInvokeRequest) -> AsyncIterator[dict[str, Any]]:
        """Emit only a terminal verified answer; planner/grader tokens and drafts stay private."""
        final: dict[str, Any] = {}
        config = self._config(request.thread_id)
        async with self._thread_lock(config):
            with agent_span("agent.stream", thread_id=request.thread_id):
                async for chunk in self._graph.astream(
                    initial_state(request.message, request.active_doc_ids),
                    config=config,
                    stream_mode="updates",
                ):
                    if not isinstance(chunk, dict):
                        continue
                    for update in chunk.values():
                        if isinstance(update, dict) and update.get("final_answer"):
                            final = update
        sources = [
            source for source in final.get("answer_sources", []) if isinstance(source, RagSource)
        ]
        if sources:
            yield {"type": "sources", "sources": [source.model_dump() for source in sources]}
        answer = str(final.get("final_answer", ""))
        for start in range(0, len(answer), 256):
            yield {"type": "delta", "text": answer[start : start + 256]}
        yield {"type": "done", "thread_id": request.thread_id, **_details(final)}

    @staticmethod
    def _config(thread_id: str | None) -> dict[str, Any]:
        # An anonymous call still needs its own checkpoint identity and must never share history.
        if not thread_id:
            from uuid import uuid4

            thread_id = str(uuid4())
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

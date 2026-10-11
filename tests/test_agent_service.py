import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from ai_agent_template.agent.service import AgentService
from ai_agent_template.api.schemas import AgentInvokeRequest
from ai_agent_template.rag.schemas import RagSource


class FakeGraph:
    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002 - match LangGraph/LangChain runnable API.
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert input["messages"][0].content == "hello"
        assert config == {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 50}
        return {"messages": [*input["messages"], AIMessage(content="world")], "sources": []}

    async def astream(
        self,
        input: dict[str, Any],  # noqa: A002 - match LangGraph/LangChain runnable API.
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        assert kwargs["stream_mode"] == "updates"
        yield {"plan": {"queries": ["private planner output"]}}
        yield {"answer": {"messages": [AIMessage(content="unverified draft")]}}
        yield {
            "verify": {
                "final_answer": "hello world",
                "answer_sources": [RagSource(source="kb", content="x")],
                "stop_reason": "answered",
                "attempts": 1,
            }
        }


@pytest.mark.asyncio
async def test_agent_service_invokes_graph() -> None:
    service = AgentService(FakeGraph())

    response = await service.invoke(AgentInvokeRequest(message="hello", thread_id="thread-1"))

    assert response.message == "world"
    assert response.thread_id == "thread-1"
    assert response.sources == []


@pytest.mark.asyncio
async def test_agent_service_streams_deltas_and_sources() -> None:
    service = AgentService(FakeGraph())

    events = [event async for event in service.stream(AgentInvokeRequest(message="hi"))]

    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert events[0]["sources"][0]["source"] == "kb"
    assert "delta" in types
    assert types[-1] == "done"
    assert "".join(event["text"] for event in events if event["type"] == "delta") == "hello world"
    assert events[-1]["retrieval_rounds"] == 1


@pytest.mark.asyncio
async def test_same_thread_requests_are_serialized() -> None:
    class ConcurrentGraph:
        active = 0
        max_active = 0

        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return {"final_answer": "done", "sources": []}

    graph = ConcurrentGraph()
    service = AgentService(graph)
    await asyncio.gather(
        service.invoke(AgentInvokeRequest(message="one", thread_id="same")),
        service.invoke(AgentInvokeRequest(message="two", thread_id="same")),
    )
    assert graph.max_active == 1

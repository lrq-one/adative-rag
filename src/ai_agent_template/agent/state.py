from typing import TypedDict

from langchain_core.messages import BaseMessage

from ai_agent_template.rag.schemas import RagSource


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    sources: list[RagSource]
    voice_samples: list[RagSource]
    queries: list[str]
    attempts: int
    verification_reason: str

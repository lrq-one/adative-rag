from functools import lru_cache

from ai_agent_template.agent.checkpointer import build_checkpointer
from ai_agent_template.agent.decisions import LLMQueryPolicy
from ai_agent_template.agent.graph import build_agent_graph
from ai_agent_template.agent.models import build_chat_model
from ai_agent_template.agent.service import AgentService
from ai_agent_template.core.settings import Settings, get_settings
from ai_agent_template.rag.evidence import LLMEvidenceVerifier
from ai_agent_template.rag.service import EmptyRagRetriever, RagService
from ai_agent_template.rag.vectorstore import build_retrieval_stores


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    settings = get_settings()
    model = build_chat_model(settings)
    query_model = build_chat_model(
        settings,
        model_name=settings.query_model_name,
        max_tokens=settings.query_model_max_tokens,
    )
    retriever = get_rag_service() if settings.rag_enabled else EmptyRagRetriever()
    checkpointer = build_checkpointer(settings)
    graph = build_agent_graph(
        model,
        retriever=retriever,
        checkpointer=checkpointer,
        max_retries=settings.rag_evidence_retry_limit,
        query_policy=LLMQueryPolicy(
            query_model,
            max_output_tokens=settings.query_model_max_tokens,
            context_max_chars=settings.rag_context_max_chars,
        ),
        max_search_calls=settings.rag_max_search_calls,
        decision_token_budget=settings.rag_decision_token_budget,
        history_turns=settings.agent_history_turns,
        context_max_chars=settings.rag_context_max_chars,
        verification_level=settings.rag_verification_level,
        evidence_verifier=LLMEvidenceVerifier(
            build_chat_model(settings, model_name=settings.rag_verifier_model_name, max_tokens=4096)
        )
        if settings.rag_verification_level == "semantic"
        else None,
        answer_repair_limit=settings.rag_answer_repair_limit,
    )
    return AgentService(graph=graph)


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    settings = get_settings()
    stores = build_retrieval_stores(settings)
    reranker = None
    if settings.rag_reranker_enabled:
        from ai_agent_template.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=settings.rag_reranker_model)
    return RagService(
        settings=settings,
        vector_store=stores["hybrid"],
        retrieval_stores=stores,
        reranker=reranker,
    )


def get_app_settings() -> Settings:
    return get_settings()

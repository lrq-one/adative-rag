"""Run the application's actual index, retrieval policy, graph, writer and verifier."""

import asyncio
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import BaseRateLimiter

from ai_agent_template.agent.decisions import Decision, EvidenceGrade, LLMQueryPolicy, QueryPlan
from ai_agent_template.agent.graph import build_agent_graph, initial_state
from ai_agent_template.agent.models import build_chat_model
from ai_agent_template.agent.state import AgentState
from ai_agent_template.core.settings import Settings
from ai_agent_template.evaluation.config import EvaluationConfig, Experiment
from ai_agent_template.evaluation.datasets import EvaluationSample, Passage
from ai_agent_template.rag.evidence import LLMEvidenceVerifier
from ai_agent_template.rag.repository import fingerprint
from ai_agent_template.rag.schemas import EvidenceRecord, RagSource
from ai_agent_template.rag.service import RagService, RetrievalMode, choose_retrieval_mode
from ai_agent_template.rag.vectorstore import build_retrieval_stores


class ExperimentalPolicy:
    def __init__(self, policy: LLMQueryPolicy, experiment: Experiment) -> None:
        self.policy, self.experiment = policy, experiment

    async def plan(
        self,
        question: str,
        history: Sequence[BaseMessage],
        active_doc_ids: Sequence[str],
        previous_sources: Sequence[RagSource],
        *,
        remaining_tokens: int,
    ) -> Decision[QueryPlan]:
        if self.experiment.route == "llm":
            return await self.policy.plan(
                question,
                history,
                active_doc_ids,
                previous_sources,
                remaining_tokens=remaining_tokens,
            )
        route = (
            choose_retrieval_mode(question)
            if self.experiment.route == "rule"
            else self.experiment.retrieval
        )
        return Decision(
            QueryPlan(
                standalone_query=question,
                intent="fact_lookup",
                route=route,
                required_facts=[question],
            )
        )

    async def grade(self, *args: Any, **kwargs: Any) -> Decision[EvidenceGrade]:
        return await self.policy.grade(*args, **kwargs)

    async def rewrite(self, *args: Any, **kwargs: Any) -> Decision[QueryPlan]:
        return await self.policy.rewrite(*args, **kwargs)


class RecordingRetriever:
    def __init__(
        self,
        rag: RagService,
        document_ids: list[str] | None = None,
        provided: list[RagSource] | None = None,
    ) -> None:
        self.rag, self.document_ids, self.provided = rag, document_ids, provided
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> list[RagSource]:
        started = time.perf_counter()
        sources = (
            self.provided
            if self.provided is not None
            else self.rag.search(query, mode=mode, source_ids=self.document_ids or source_ids)
        )
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "sources": [source.model_dump() for source in sources],
            }
        )
        return sources

    def search_voice(self, query: str) -> list[RagSource]:
        return []

    def evidence_status(self, record: EvidenceRecord) -> str:
        return self.rag.evidence_status(record)


class ProjectExecutor:
    def __init__(
        self,
        settings: Settings,
        config: EvaluationConfig,
        experiment: Experiment,
        samples: list[EvaluationSample],
        directory: Path,
    ) -> None:
        self.config, self.experiment = config, experiment
        storage = (
            Path(config.storage_root) / directory.parent.name / directory.name
            if config.storage_root
            else directory
        )
        self.settings = settings.model_copy(
            update={
                "rag_data_path": str(storage / "index"),
                "qdrant_url": None,
                "qdrant_storage_path": str(storage / "qdrant"),
                "rag_collection_name": "evaluation",
                "rag_top_k": config.top_k,
                "rag_search_multiplier": config.candidate_multiplier,
                "rag_chunk_strategy": experiment.chunking,
                "rag_embedding_model_name": experiment.embedding_model,
                "rag_embedding_dimensions": experiment.embedding_dimensions,
                "rag_verification_level": experiment.verification,
                "rag_reranker_enabled": experiment.reranker,
                "rag_namespace": "default",
            }
        )
        import json

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "effective_settings.json").write_text(
            json.dumps(
                self.settings.model_dump(
                    exclude={"openai_api_key", "groq_api_key", "qdrant_api_key", "auth_api_keys"}
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stores = build_retrieval_stores(self.settings)
        reranker = None
        if experiment.reranker:
            from ai_agent_template.rag.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(self.settings.rag_reranker_model)
        self.rag = RagService(
            self.settings, stores["hybrid"], retrieval_stores=stores, reranker=reranker
        )
        self.client = stores["hybrid"].client
        documents: defaultdict[str, dict[str, Passage]] = defaultdict(dict)
        for sample in samples:
            for passage in sample.passages:
                documents[passage.document_key][passage.unit_id] = passage
        self.document_ids: dict[str, str] = {}
        for key, units in documents.items():
            sections = [
                Document(
                    page_content=unit.text,
                    metadata={
                        "section": unit.section,
                        "dataset_unit_id": unit.unit_id,
                        "dataset_document_key": key,
                    },
                )
                for unit in units.values()
            ]
            result = self.rag.ingest_sections(
                sections=sections, source=key + ".txt", document_id=key
            )
            assert result.document_id is not None
            self.document_ids[key] = result.document_id
        self.model = None
        self.policy = None
        self.limiter: BaseRateLimiter | None = None
        if config.phase != "retrieval" or experiment.route == "llm":
            self.model = build_chat_model(self.settings)
            self.policy = ExperimentalPolicy(
                LLMQueryPolicy(
                    build_chat_model(
                        self.settings, model_name=self.settings.query_model_name, max_tokens=1200
                    )
                ),
                experiment,
            )

    def set_rate_limiter(self, limiter: BaseRateLimiter) -> None:
        self.limiter = limiter
        if self.model:
            self.model.rate_limiter = limiter
        if self.policy:
            self.policy.policy._model.rate_limiter = limiter

    async def run_sample(self, sample: EvaluationSample) -> dict[str, Any]:
        doc_ids = [
            self.document_ids[key] for key in dict.fromkeys(p.document_key for p in sample.passages)
        ]
        scope = doc_ids if sample.dataset == "qasper" else None
        provided = None
        if self.config.context_mode == "provided":
            provided = []
            for doc_id in doc_ids:
                manifest = self.rag.repository.active(doc_id)
                assert manifest is not None
                for chunk_id in manifest.evidence_ids or manifest.chunk_ids:
                    resolved = self.rag.repository.resolve(chunk_id)
                    if not resolved:
                        raise RuntimeError("Provided evidence became unavailable")
                    record, _ = resolved
                    provided.append(
                        RagSource(
                            source=record.source_filename,
                            content=record.text,
                            metadata=record.metadata,
                            evidence=record,
                        )
                    )
        telemetry = RecordingRetriever(self.rag, scope, provided)
        started = time.perf_counter()
        if self.config.phase == "retrieval":
            tokens = 0
            if self.experiment.route == "llm":
                assert self.policy is not None
                decision = await self.policy.plan(
                    sample.question, [], [], [], remaining_tokens=16000
                )
                if decision.value is None or decision.value.needs_clarification:
                    return {
                        "answer": None,
                        "claims": [],
                        "citations": [],
                        "retrieved": [],
                        "retrieval_calls": [],
                        "stop_reason": "clarification_required"
                        if decision.value
                        else decision.error,
                        "decision_tokens": decision.tokens,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                queries = decision.value.sub_queries or [decision.value.standalone_query]
                mode, tokens = decision.value.route, decision.tokens
            else:
                mode = (
                    choose_retrieval_mode(sample.question)
                    if self.experiment.route == "rule"
                    else self.experiment.retrieval
                )
                queries = [sample.question]
            for query in queries[:3]:
                await asyncio.to_thread(telemetry.search, query, mode=mode)
            state: AgentState = {
                "final_answer": "",
                "claims": [],
                "citations": [],
                "stop_reason": "retrieval_only",
                "decision_tokens": tokens,
            }
        else:
            assert self.model is not None and self.policy is not None
            verifier_model = build_chat_model(
                self.settings, model_name=self.settings.rag_verifier_model_name, max_tokens=4096
            )
            verifier_model.rate_limiter = self.limiter
            graph = build_agent_graph(
                self.model,
                telemetry,
                query_policy=self.policy,
                max_retries=2 if self.experiment.feedback and provided is None else 0,
                verification_level=self.experiment.verification,
                evidence_verifier=LLMEvidenceVerifier(verifier_model)
                if self.experiment.verification == "semantic"
                else None,
                max_sources=self.config.top_k,
                max_search_calls=self.settings.rag_max_search_calls,
                decision_token_budget=self.settings.rag_decision_token_budget,
                history_turns=self.settings.agent_history_turns,
                context_max_chars=self.settings.rag_context_max_chars,
                answer_repair_limit=self.settings.rag_answer_repair_limit,
            )
            state = cast(
                AgentState,
                await graph.ainvoke(initial_state(sample.question), config={"recursion_limit": 60}),
            )
        unique: dict[str, dict[str, Any]] = {}
        for call in telemetry.calls:
            for source in call["sources"]:
                unique.setdefault(source["evidence"]["evidence_id"], source)
        return {
            "answer": state["final_answer"] if self.config.phase != "retrieval" else None,
            "claims": [claim.model_dump() for claim in state.get("claims", [])],
            "citations": [citation.model_dump() for citation in state.get("citations", [])],
            "verification": state["verification"].model_dump()
            if state.get("verification")
            else None,
            "retrieved": list(unique.values()),
            "retrieval_calls": telemetry.calls,
            "decision_tokens": state.get("decision_tokens", 0),
            "verification_tokens": state.get("verification_tokens", 0),
            "generation_tokens": state.get("generation_tokens", 0),
            "verification_calls": state.get("verification_calls", 0),
            "answer_repairs": state.get("answer_repairs", 0),
            "stop_reason": state.get("stop_reason", ""),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "index_snapshot": fingerprint(
                [item.model_dump() for item in self.rag.repository.snapshot()]
            ),
        }

    def close(self) -> None:
        self.client.close()

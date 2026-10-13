"""Reuse versioned parsing, evidence and bounded Adaptive-RAG.

Disconnected demo uses actual lexical BM25 over persisted evidence, with no fake LLM
or fake dense encoder. Production uses existing Qdrant hybrid and optional reranker.
"""

import asyncio
import math
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import UploadFile
from langchain_core.documents import Document

from ai_agent_template.agent.decisions import LLMQueryPolicy, QueryPlan
from ai_agent_template.commerce.operations_models import AnalysisRequest, SupplierDocument
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.commerce.service import CommerceService
from ai_agent_template.rag.loaders import load_upload_documents, parser_identity, save_original
from ai_agent_template.rag.repository import EvidenceRepository, digest, fingerprint
from ai_agent_template.rag.schemas import IndexStats, RagSource
from ai_agent_template.rag.service import RagService


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", text.casefold())


class PersistedBM25:
    """No vector simulation: lexical BM25 computes scores from active saved chunks."""

    supports_source_ids = True

    def __init__(self, repository: EvidenceRepository, namespace: str) -> None:
        self.repository, self.namespace = repository, namespace

    def add_documents(self, documents: list[Document], **kwargs: Any) -> None:
        # VersionedIndexer publishes the immutable chunks in EvidenceRepository.
        pass

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        evidence = [
            resolved[0]
            for manifest in self.repository.snapshot(self.namespace)
            if not kwargs.get("source_ids") or manifest.document_id in kwargs["source_ids"]
            for eid in manifest.evidence_ids
            if (resolved := self.repository.resolve(eid))
        ]
        counts = [Counter(tokens(item.text)) for item in evidence]
        lengths = [sum(count.values()) for count in counts]
        average = sum(lengths) / max(1, len(lengths))
        scores = []
        for item, count, length in zip(evidence, counts, lengths, strict=True):
            score = 0.0
            for term in set(tokens(query)):
                df = sum(term in candidate for candidate in counts)
                tf = count[term]
                idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(1, average)))
            if score > 0:
                scores.append((Document(page_content=item.text, metadata=item.metadata), score))
        return sorted(scores, key=lambda pair: pair[1], reverse=True)[:k]


class OperationsKnowledge:
    def __init__(self, commerce: CommerceService, repository: OperationsRepository) -> None:
        self.commerce = commerce
        self.repository = repository
        self._offline: dict[str, RagService] = {}

    def rag(self, request: AnalysisRequest) -> RagService:
        key = fingerprint([request.seller_id, request.marketplace_id, request.sku, request.asin])
        if self.commerce.live:
            return self.commerce.rag("operations:" + key)
        if self.commerce.settings.commerce_production:
            raise ValueError("Production knowledge requires real embedding configuration")
        if key not in self._offline:
            path = (
                Path(self.commerce.settings.rag_data_path).parent
                / ".commerce"
                / "operations-knowledge"
                / key[:16]
            )
            repository = EvidenceRepository(path / "manifest.sqlite3")
            settings = self.commerce.settings.model_copy(
                update={
                    "rag_data_path": str(path),
                    "rag_namespace": key,
                    "rag_embedding_model_name": "lexical-bm25-no-dense-v1",
                    "rag_retrieval_mode": "sparse",
                }
            )
            store = PersistedBM25(repository, key)
            self._offline[key] = RagService(
                settings, store, retrieval_stores={"sparse": store}, repository=repository
            )
        return self._offline[key]

    async def upload(
        self,
        request: AnalysisRequest,
        content: bytes,
        filename: str,
        product_version: str,
        source_key: str | None = None,
    ) -> dict[str, Any]:
        rag = self.rag(request)
        stats = IndexStats()
        sections = await load_upload_documents(
            UploadFile(file=BytesIO(content), filename=filename),
            rag._settings,
            rag.indexer.cache,
            stats,
        )
        result = await asyncio.to_thread(
            rag.ingest_sections,
            sections=sections,
            source=filename,
            metadata={
                "seller_id": request.seller_id,
                "marketplace_id": request.marketplace_id,
                "sku": request.sku,
                "asin": request.asin,
                "product_version": product_version,
                "effective_at": request.start.isoformat(),
            },
            document_id=source_key,
            namespace=rag._settings.rag_namespace,
            file_hash=digest(content),
            original_file=save_original(
                content, Path(filename).suffix, rag._settings.rag_data_path
            ),
            parser_version=parser_identity(rag._settings),
            preprocessing_stats=stats,
        )
        return result.model_dump(mode="json")

    async def retrieve(
        self,
        request: AnalysisRequest,
        queries: list[str],
        route: Literal["dense", "sparse", "hybrid"],
        model: Any = None,
        decision_budget: int | None = None,
    ) -> dict[str, Any]:
        rag = self.rag(request)
        allowed = []
        owned_versions = {
            (item.document_id, item.document_version)
            for item in self.repository.list(
                SupplierDocument, request.seller_id, request.marketplace_id, request.sku
            )
            if item.document_id
            and (not request.asin or item.asin == request.asin)
            and item.effective_at <= request.end
            and (not item.expires_at or item.expires_at > request.end)
        }
        for manifest in rag.repository.snapshot(rag._settings.rag_namespace):
            if (manifest.document_id, manifest.document_version) not in owned_versions:
                continue
            resolved = (
                rag.repository.resolve(manifest.evidence_ids[0]) if manifest.evidence_ids else None
            )
            meta = resolved[0].metadata if resolved else {}
            if meta.get("effective_at", "0001-01-01") <= request.end.isoformat() and (
                not request.product_version
                or meta.get("product_version") == request.product_version
            ):
                allowed.append(manifest.document_id)
        if not allowed:
            return dict(
                sources=[],
                search_calls=0,
                decision_tokens=0,
                attempted_queries=[],
                evidence_grade=None,
                mode="qdrant_hybrid" if self.commerce.live else "offline_lexical_bm25",
                missing_evidence=["无所选版本/有效日期范围的相关文档；历史原文仍可按版本查看"],
            )
        sources: list[RagSource] = []
        search_calls = 0
        decision_tokens = 0
        grade = None
        grade_error = ""
        budget = self.commerce.settings.rag_max_search_calls
        token_budget = (
            decision_budget
            if decision_budget is not None
            else self.commerce.settings.rag_decision_token_budget
        )
        attempted = []
        selected_route: Literal["dense", "sparse", "hybrid"] = (
            route if self.commerce.live else "sparse"
        )
        for query in queries[:3]:
            if search_calls >= budget:
                break
            found = await asyncio.to_thread(
                rag.search, query, mode=selected_route, source_ids=allowed
            )
            sources.extend(found)
            attempted.append(query)
            search_calls += 1
        policy = LLMQueryPolicy(model) if model is not None else None
        plan = QueryPlan(
            standalone_query=queries[0] if queries else request.sku,
            intent="multi_hop",
            route=selected_route,
            required_facts=queries[:3],
        )

        def unique() -> list[RagSource]:
            return list(
                {
                    source.evidence.evidence_id: source
                    for source in sources
                    if source.evidence and source.evidence.metadata.get("sku") == request.sku
                }.values()
            )

        if policy and sources:
            assessed = await policy.grade(plan, unique(), remaining_tokens=token_budget)
            decision_tokens += assessed.tokens
            grade = assessed.value
            grade_error = assessed.error
            retries = 0
            while (
                grade
                and not grade.sufficient
                and retries < self.commerce.settings.rag_evidence_retry_limit
                and search_calls < budget
            ):
                rewrite = await policy.rewrite(
                    plan,
                    unique(),
                    grade,
                    attempted,
                    remaining_tokens=token_budget - decision_tokens,
                )
                decision_tokens += rewrite.tokens
                if not rewrite.value or rewrite.value.standalone_query in attempted:
                    break
                plan = rewrite.value
                attempted.append(plan.standalone_query)
                sources.extend(
                    await asyncio.to_thread(
                        rag.search,
                        plan.standalone_query,
                        mode=plan.route if self.commerce.live else "sparse",
                        source_ids=allowed,
                    )
                )
                search_calls += 1
                retries += 1
                assessed = await policy.grade(
                    plan,
                    unique(),
                    remaining_tokens=token_budget - decision_tokens,
                )
                decision_tokens += assessed.tokens
                grade = assessed.value
                grade_error = assessed.error
        return dict(
            sources=[source.model_dump(mode="json") for source in unique()],
            search_calls=search_calls,
            decision_tokens=decision_tokens,
            attempted_queries=attempted,
            evidence_grade=grade.model_dump() if grade else None,
            decision_error=grade_error,
            mode="qdrant_hybrid" if self.commerce.live else "offline_lexical_bm25",
            missing_evidence=grade.missing_facts
            if grade
            else (
                ["检索证据覆盖度未能通过模型评估：" + grade_error]
                if grade_error
                else ([] if sources else ["无相关文档证据"])
            ),
        )

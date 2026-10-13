import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ai_agent_template.agent.decisions import QueryPlan
from ai_agent_template.agent.graph import build_agent_graph
from ai_agent_template.agent.models import build_chat_model
from ai_agent_template.agent.service import AgentService
from ai_agent_template.api.schemas import AgentInvokeRequest
from ai_agent_template.commerce.demo import manual, seed_structured
from ai_agent_template.commerce.errors import CommerceModelUnavailable
from ai_agent_template.commerce.importers import read_catalog
from ai_agent_template.commerce.listing import generate
from ai_agent_template.commerce.offline import demo_stores
from ai_agent_template.commerce.orders import MockOrderAdapter, active_policy
from ai_agent_template.commerce.repository import CommerceRepository
from ai_agent_template.commerce.retrieval import (
    CommerceLLMPolicy,
    CommerceRetriever,
    CommerceScope,
    PlannedPolicy,
    scope,
)
from ai_agent_template.commerce.schemas import (
    CatalogImport,
    ChatRequest,
    ListingDraft,
    ListingRequest,
    Principal,
    Product,
    ProductDocument,
    ProductVariant,
    validate_dates,
)
from ai_agent_template.commerce.verification import CommerceVerifier
from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.evidence import LLMEvidenceVerifier
from ai_agent_template.rag.indexing import logical_id
from ai_agent_template.rag.loaders import load_upload_documents, parser_identity, save_original
from ai_agent_template.rag.repository import digest
from ai_agent_template.rag.schemas import IndexStats
from ai_agent_template.rag.service import RagService
from ai_agent_template.rag.vectorstore import build_retrieval_stores


class CommerceService:
    def __init__(self, settings: Settings, repository: CommerceRepository | None = None) -> None:
        self.settings = settings
        self.repository = repository or CommerceRepository(
            settings.commerce_database_url, production=settings.commerce_production
        )
        if not settings.commerce_production and settings.commerce_seed_demo:
            seed_structured(self.repository)
        self._stores: dict[str, Any] | None = None
        self._rag_lock = RLock()
        self._rags: dict[str, RagService] = {}
        self._agent: AgentService | None = None
        self._model: Any = None
        self._policy: Any = None
        self._judge: Any = None
        self._locks: dict[str, asyncio.Lock] = {}
        self.retriever = CommerceRetriever(self.repository, self.rag)
        self.orders = MockOrderAdapter(self.repository)

    @property
    def live(self) -> bool:
        return self.settings.commerce_llm_mode == "live"

    def rag(self, store: str) -> RagService:
        with self._rag_lock:
            return self._rag(store)

    def _rag(self, store: str) -> RagService:
        if self.settings.commerce_production and not self.live:
            raise CommerceModelUnavailable("生产知识检索需要配置真实模型，不能使用离线测试向量。")
        if self._stores is None:
            path = Path(self.settings.rag_data_path).parent / ".commerce" / "knowledge"
            # A distinct path/collection prevents reuse of incompatible historical research vectors.
            settings = self.settings.model_copy(
                update={
                    "rag_data_path": str(path),
                    "qdrant_storage_path": str(path / "qdrant"),
                    "rag_collection_name": "commerce_knowledge",
                    "rag_embedding_model_name": self.settings.commerce_embedding_model,
                    "rag_embedding_dimensions": self.settings.commerce_embedding_dimensions,
                }
            )
            self._stores = build_retrieval_stores(settings) if self.live else demo_stores(settings)
            self._rag_settings = settings
        if store not in self._rags:
            settings = self._rag_settings.model_copy(update={"rag_namespace": "commerce:" + store})
            reranker = None
            if self.live and settings.rag_reranker_enabled:
                from ai_agent_template.rag.rerank import CrossEncoderReranker

                reranker = CrossEncoderReranker(settings.rag_reranker_model)
            self._rags[store] = RagService(
                settings, self._stores["hybrid"], retrieval_stores=self._stores, reranker=reranker
            )
        return self._rags[store]

    def models(self) -> None:
        if self.settings.commerce_production and not self.live:
            raise CommerceModelUnavailable("生产环境需要配置真实 LLM，离线演示不作为生产模型。")
        if self.live:
            credential = (
                self.settings.openai_api_key
                if self.settings.model_provider == "openai"
                else self.settings.groq_api_key
            )
            if not credential or not credential.get_secret_value():
                raise CommerceModelUnavailable("真实 LLM 未配置：需要模型服务凭证")
        if self._policy is not None:
            return
        if self.live:
            key = (
                self.settings.openai_api_key
                if self.settings.model_provider == "openai"
                else self.settings.groq_api_key
            )
            if not key or not key.get_secret_value():
                raise CommerceModelUnavailable(
                    "真实 LLM 未配置：请设置模型 API Key、Base URL 和模型名称，或选择明确的 offline Demo 模式"
                )
            self._model = build_chat_model(self.settings)
            self._policy = CommerceLLMPolicy(
                build_chat_model(self.settings, model_name=self.settings.query_model_name),
                self.repository,
                max_output_tokens=self.settings.query_model_max_tokens,
            )
            self._judge = CommerceVerifier(
                self.repository,
                LLMEvidenceVerifier(
                    build_chat_model(
                        self.settings, model_name=self.settings.rag_verifier_model_name
                    )
                ),
            )
        else:
            raise CommerceModelUnavailable(
                "离线模式不使用 Fake LLM；请使用运营确定性分析或配置真实模型。"
            )

    def agent(self) -> AgentService:
        self.models()
        if self._agent is None:
            self._agent = AgentService(
                build_agent_graph(
                    self._model,
                    self.retriever,
                    MemorySaver(),
                    query_policy=PlannedPolicy(self._policy),
                    max_retries=self.settings.rag_evidence_retry_limit,
                    max_search_calls=self.settings.rag_max_search_calls,
                    decision_token_budget=self.settings.rag_decision_token_budget,
                    history_turns=self.settings.agent_history_turns,
                    verification_level="semantic" if self.live else "deterministic",
                    evidence_verifier=self._judge,
                    answer_repair_limit=self.settings.rag_answer_repair_limit,
                )
            )
        return self._agent

    def catalog(self, principal: Principal, request: CatalogImport) -> dict[str, int]:
        items = [*request.products, *request.variants]
        if any(item.store_id != principal.store_id for item in items):
            raise ValueError("Catalog belongs to another store")
        known = {item.id for item in self.repository.list(Product, principal.store_id)} | {
            item.id for item in request.products
        }
        if any(
            item.product_id not in known or item.id != item.sku for item in request.variants
        ) or len({item.sku for item in request.variants}) != len(request.variants):
            raise ValueError("Variants need unique SKU ids and existing products")
        self.repository.put_many(items)
        return {"products": len(request.products), "variants": len(request.variants)}

    async def catalog_file(
        self, principal: Principal, content: bytes, filename: str
    ) -> dict[str, Any]:
        request, descriptions = await asyncio.to_thread(
            read_catalog, content, filename, principal.store_id
        )
        counts = await asyncio.to_thread(self.catalog, principal, request)
        results = []
        for row in descriptions:
            results.append(
                await self.upload(
                    principal,
                    row["sku"],
                    row["text"].encode(),
                    row["sku"] + "-catalog.md",
                    row["market"],
                    row["language"],
                )
            )
        return {**counts, "documents": results}

    async def upload(
        self,
        principal: Principal,
        sku: str,
        content: bytes,
        filename: str,
        market: str,
        language: str,
        document_id: str | None = None,
        effective_at: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        validate_dates(effective_at, expires_at)
        variant = self.repository.get(ProductVariant, principal.store_id, sku)
        if not variant:
            raise LookupError("SKU not found")
        if variant.markets and market not in variant.markets:
            raise ValueError("SKU not offered in this market")
        if document_id:
            prior = self.repository.get(ProductDocument, principal.store_id, document_id)
            if not prior or prior.sku != sku:
                raise LookupError("Document not found for this SKU/store")
        product = self.repository.get(Product, principal.store_id, variant.product_id)
        rag = await asyncio.to_thread(self.rag, principal.store_id)
        namespace = "commerce:" + principal.store_id
        owner = document_id or logical_id(namespace, sku + ":" + filename, "knowledge")
        stats = IndexStats()
        with rag.indexer.cache.bind(owner):
            sections = await load_upload_documents(
                UploadFile(file=BytesIO(content), filename=filename),
                rag._settings,
                rag.indexer.cache,
                stats,
            )
        metadata = {
            "store_id": principal.store_id,
            "product_id": variant.product_id,
            "sku": sku,
            "category": product.category if product else "",
            "market": market,
            "language": language,
            "effective_at": effective_at,
            "expires_at": expires_at,
        }
        result = await asyncio.to_thread(
            rag.ingest_sections,
            sections=sections,
            source=filename,
            metadata=metadata,
            namespace=namespace,
            document_id=owner,
            file_hash=digest(content),
            original_file=save_original(
                content, Path(filename).suffix, rag._settings.rag_data_path
            ),
            parser_version=parser_identity(rag._settings),
            preprocessing_stats=stats,
        )
        manifest = rag.repository.versions(str(result.document_id))[-1]
        item = ProductDocument(
            id=manifest.document_id,
            store_id=principal.store_id,
            product_id=variant.product_id,
            sku=sku,
            category=metadata["category"],
            market=market,
            language=language,
            source=filename,
            document_id=manifest.document_id,
            document_version=manifest.document_version,
            evidence_ids=manifest.evidence_ids,
            effective_at=effective_at,
            expires_at=expires_at,
            synthetic=variant.synthetic,
        )
        self.repository.put(item)
        return {"document": item.model_dump(), "index": result.model_dump()}

    async def seed_knowledge(self, principal: Principal) -> dict[str, int]:
        if self.settings.commerce_mode != "demo" or principal.store_id != "demo-store":
            raise ValueError("Demo seeding is disabled for this store/environment")
        count = 0
        for variant in self.repository.list(ProductVariant, principal.store_id):
            product = self.repository.get(Product, principal.store_id, variant.product_id)
            if product:
                await self.upload(
                    principal,
                    variant.sku,
                    manual(variant, product).encode(),
                    variant.sku + ".md",
                    "US",
                    "multilingual",
                )
                count += 1
        return {"documents": count}

    async def listing(self, principal: Principal, request: ListingRequest) -> ListingDraft:
        variant = self.repository.get(ProductVariant, principal.store_id, request.sku)
        product = (
            self.repository.get(Product, principal.store_id, variant.product_id)
            if variant
            else None
        )
        if not variant or not product:
            raise LookupError("SKU not found")
        if variant.markets and request.market not in variant.markets:
            raise ValueError("SKU is not offered in the requested market")
        self.models()
        token = scope.set(CommerceScope(principal, request.market, request.language, [request.sku]))
        try:
            sources = await asyncio.to_thread(self.retriever.search, request.sku, mode="hybrid")
            content, issues, facts = await generate(
                request,
                variant,
                product,
                sources,
                self._model,
                self._judge,
                self.retriever.evidence_status,
                self.live,
                self.settings.commerce_listing_rules,
            )
        finally:
            scope.reset(token)
        draft = ListingDraft(
            id=str(uuid4()),
            store_id=principal.store_id,
            sku=request.sku,
            platform=request.platform,
            market=request.market,
            language=request.language,
            title=content.title if "title" in request.content_types else "",
            bullets=content.bullets if "bullets" in request.content_types else [],
            description=content.description if "description" in request.content_types else "",
            facts=facts,
            issues=issues,
            review_status="blocked"
            if any("校验未通过" in issue or "未绑定" in issue for issue in issues)
            else "pending_review",
            generation_mode="live_llm" if self.live else "offline_extractive_demo",
            synthetic=variant.synthetic,
        )
        self.repository.put(draft)
        return draft

    async def chat(self, principal: Principal, request: ChatRequest) -> dict[str, Any]:
        key = digest(principal.model_dump_json() + ":" + request.thread_id)
        lock = self._locks.setdefault(principal.store_id + key, asyncio.Lock())
        async with lock:
            return await self._chat(principal, request, key)

    async def _chat(self, principal: Principal, request: ChatRequest, key: str) -> dict[str, Any]:
        self.models()
        saved = self.repository.session(principal.store_id, key)
        selected = request.skus or saved.get("skus", [])
        current = CommerceScope(principal, request.market, request.language, selected)
        token = scope.set(current)
        try:
            history = [
                message
                for item in saved.get("turns", [])[-4:]
                for message in (
                    HumanMessage(content=item["question"]),
                    AIMessage(content=item.get("answer", "")[:1000]),
                )
            ]
            decision = await self._policy.plan(
                request.message,
                history,
                [],
                [],
                remaining_tokens=self.settings.rag_decision_token_budget,
            )
            if not decision.value:
                raise ValueError("问题理解失败：" + decision.error)
            plan: QueryPlan = decision.value
            current.plan = plan
            current.decision_tokens = decision.tokens
            current.skus = plan.skus or selected
            plan.entities = list(dict.fromkeys([*plan.entities, *current.skus]))[:12]
            if decision.tokens >= self.settings.rag_decision_token_budget:
                raise ValueError("Query planning token budget exhausted")
            if any(
                not self.repository.get(ProductVariant, principal.store_id, sku)
                for sku in current.skus
            ):
                raise LookupError("SKU not found in this store")
            if plan.needs_clarification:
                result: dict[str, Any] = {
                    "answer": plan.clarification_question,
                    "stop_reason": "clarification_required",
                    "citations": [],
                }
            elif plan.intent == "order_status":
                order_id = request.order_id
                if not order_id:
                    result = {
                        "answer": "请提供有权限访问的订单编号。",
                        "stop_reason": "clarification_required",
                        "citations": [],
                    }
                else:
                    data = await asyncio.to_thread(self.orders.query, principal, order_id)
                    labels = {"en": "Order status", "es": "Estado del pedido", "zh": "订单状态"}
                    result = {
                        "answer": labels.get(request.language, "Order status")
                        + ": "
                        + data["order"]["status"]
                        + "; "
                        + data["order"]["fulfillment_status"],
                        "order_data": data,
                        "citations": [],
                        "stop_reason": "authoritative_business_data",
                    }
            elif plan.intent == "listing_generation" and current.skus:
                if principal.role != "operator":
                    raise PermissionError("Listing drafts require operator access")
                draft = await self.listing(
                    principal,
                    ListingRequest(
                        sku=current.skus[0],
                        platform=plan.platform or "amazon",
                        market=request.market,
                        language=request.language,
                    ),
                )
                result = {
                    "answer": draft.title + "\n\n" + draft.description,
                    "listing": draft.model_dump(),
                    "citations": [],
                    "stop_reason": "draft_pending_review",
                }
            elif plan.intent in {"return_refund", "complaint", "shipping_policy"}:
                policy = active_policy(
                    self.repository,
                    principal.store_id,
                    request.market,
                    "shipping" if plan.intent == "shipping_policy" else "return",
                )
                policy_order_data = (
                    await asyncio.to_thread(self.orders.query, principal, request.order_id)
                    if request.order_id
                    else None
                )
                response = await self.agent().invoke(
                    AgentInvokeRequest(
                        message=request.message + f"\nAnswer language: {request.language}",
                        thread_id=principal.store_id + ":" + key,
                        active_doc_ids=[],
                    )
                )
                result = {
                    **response.model_dump(),
                    "policy": policy.model_dump() if policy else None,
                    "order_data": policy_order_data,
                    "business_source": {
                        "adapter": "active_store_policy",
                        "record_id": policy.id if policy else None,
                        "updated_at": policy.updated_at if policy else None,
                    },
                    "stop_reason": "policy_read_only",
                }
                if request.order_id and plan.after_sales_reason:
                    from ai_agent_template.commerce.orders import after_sales
                    from ai_agent_template.commerce.schemas import AfterSalesRequest

                    result["after_sales"] = await asyncio.to_thread(
                        after_sales,
                        self.repository,
                        principal,
                        request.order_id,
                        AfterSalesRequest(reason=plan.after_sales_reason),
                    )
            else:
                agent_request = AgentInvokeRequest(
                    message=request.message + f"\nAnswer language: {request.language}",
                    thread_id=principal.store_id + ":" + key,
                    active_doc_ids=[],
                )
                response = await self.agent().invoke(agent_request)
                result = response.model_dump()
            result.update(
                {
                    "query_plan": plan.model_dump(),
                    "mode": "live_llm" if self.live else "offline_demo",
                    "thread_id": request.thread_id,
                }
            )
            turns = [
                *saved.get("turns", []),
                {"question": request.message, "answer": result.get("answer", "")},
            ][-4:]
            self.repository.save_session(
                principal.store_id, key, {"skus": current.skus, "turns": turns}
            )
            return result
        except Exception as error:
            from groq import APIError as GroqAPIError
            from openai import APIError

            if isinstance(error, (APIError, GroqAPIError, TimeoutError, ConnectionError)):
                raise CommerceModelUnavailable(
                    "模型服务暂时不可用，请检查配置或稍后重试。"
                ) from error
            raise
        finally:
            scope.reset(token)

    async def stream(
        self, principal: Principal, request: ChatRequest
    ) -> AsyncIterator[dict[str, Any]]:
        result = await self.chat(principal, request)
        # All tools/verification finish before the first answer delta, as in the original SSE.
        yield {"type": "sources", "sources": result.get("sources", [])}
        text = result.get("answer") or ""
        for start in range(0, len(text), 256):
            yield {"type": "delta", "text": text[start : start + 256]}
        yield {"type": "done", **result}

    def close(self) -> None:
        if self._stores:
            self._stores["hybrid"].client.close()
        self.repository.engine.dispose()

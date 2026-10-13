"""Explicit synthetic offline adapters. Not a semantic LLM or production embedding model."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from ai_agent_template.agent.decisions import Decision, EvidenceGrade, QueryPlan
from ai_agent_template.commerce.repository import CommerceRepository
from ai_agent_template.commerce.schemas import ProductVariant
from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.cache import CachedEmbeddings, CachedSparseEmbeddings, PersistentCache
from ai_agent_template.rag.schemas import RagSource
from ai_agent_template.rag.vectorstore import ensure_collection


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower()) + list(re.findall(r"[\u4e00-\u9fff]", text))


class DemoDense(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for token in tokens(text):
            vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 64] += 1
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class DemoSparse(SparseEmbeddings):
    def embed_query(self, text: str) -> SparseVector:
        counts = Counter(
            int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) for token in tokens(text)
        )
        return SparseVector(
            indices=list(counts), values=[float(value) for value in counts.values()]
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.embed_query(text) for text in texts]


def demo_stores(settings: Settings) -> dict[str, QdrantVectorStore]:
    client = (
        QdrantClient(url=settings.qdrant_url)
        if settings.qdrant_url
        else QdrantClient(path=str(Path(settings.rag_data_path) / "demo-qdrant"))
    )
    name = "commerce_demo_hash_v1"
    ensure_collection(
        client,
        settings.model_copy(update={"rag_collection_name": name, "rag_embedding_dimensions": 64}),
    )
    cache = PersistentCache(Path(settings.rag_data_path) / "cache.sqlite3")
    common: dict[str, Any] = dict(
        client=client,
        collection_name=name,
        vector_name="dense",
        sparse_vector_name="sparse",
        embedding=CachedEmbeddings(DemoDense(), cache, "OFFLINE-hash-v1", 64),
        sparse_embedding=CachedSparseEmbeddings(DemoSparse(), cache, "OFFLINE-sparse-v1"),
    )
    return {
        mode: QdrantVectorStore(**common, retrieval_mode=getattr(RetrievalMode, mode.upper()))
        for mode in ("dense", "sparse", "hybrid")
    }


class FakeCommercePolicy:
    """Only the explicitly selected offline mode uses keyword fixture routing."""

    def __init__(self, repository: CommerceRepository) -> None:
        self.repository = repository

    async def plan(
        self,
        question: str,
        history: Sequence[BaseMessage],
        active_doc_ids: Sequence[str],
        previous_sources: Sequence[RagSource],
        *,
        remaining_tokens: int,
    ) -> Decision[QueryPlan]:
        lower = question.lower()
        intent = "product_faq"
        for words, value in (
            (r"compare|比较|对比|comparar", "product_comparison"),
            (r"listing|标题|卖点", "listing_generation"),
            (r"return|refund|退货|退款|devoluci", "return_refund"),
            (r"order|订单|pedido", "order_status"),
            (r"shipping|配送|envío", "shipping_policy"),
            (r"complaint|投诉", "complaint"),
        ):
            if re.search(words, lower):
                intent = value
                break
        skus = re.findall(r"\bDEMO-[A-Z]+-\d+\b", question.upper())
        from ai_agent_template.commerce.retrieval import scope

        current = scope.get()
        known = self.repository.list(ProductVariant, current.principal.store_id)
        explicit = [item.sku for item in known if item.sku.casefold() in lower]
        skus = explicit or skus
        if not skus and history:
            skus = re.findall(
                r"\bDEMO-[A-Z]+-\d+\b", " ".join(str(item.content) for item in history).upper()
            )[-1:]
        skus = skus or current.skus
        if len(skus) == 1 and re.search(r"小容量|small|pequeñ", lower):
            variant = self.repository.get(ProductVariant, current.principal.store_id, skus[0])
            siblings = [item for item in known if variant and item.product_id == variant.product_id]
            capacities = [
                (float(match.group()), item.sku)
                for item in siblings
                if (match := re.search(r"\d+(?:\.\d+)?", item.attributes.get("capacity", "")))
            ]
            if capacities:
                skus = [min(capacities)[1]]
        language = (
            "zh"
            if re.search(r"[\u4e00-\u9fff]", question)
            else "es"
            if re.search(r"cuál|capacidad|contiene|pedido|comparar", lower)
            else "en"
        )
        return Decision(
            QueryPlan(
                standalone_query=question,
                original_query=question,
                intent=intent,
                route="hybrid",
                skus=list(dict.fromkeys(skus)),
                sub_queries=[sku + " " + question for sku in skus[:3]]
                if intent == "product_comparison"
                else [],
                needs_clarification=not skus
                and intent in {"product_faq", "product_comparison", "listing_generation"},
                clarification_question="请提供商品 SKU 或选择具体变体。"
                if not skus
                and intent in {"product_faq", "product_comparison", "listing_generation"}
                else None,
                language=language,
                required_facts=[question],
            )
        )

    async def grade(
        self, plan: QueryPlan, sources: Sequence[RagSource], *, remaining_tokens: int
    ) -> Decision[EvidenceGrade]:
        # Extractive demonstration only: never label this as semantic verification.
        return Decision(
            EvidenceGrade(
                sufficient=bool(sources),
                relevant_indices=list(range(min(12, len(sources)))),
                covered_facts=plan.required_facts if sources else [],
                missing_facts=[] if sources else plan.required_facts,
            )
        )

    async def rewrite(
        self,
        plan: QueryPlan,
        sources: Sequence[RagSource],
        grade: EvidenceGrade,
        attempted_queries: Sequence[str],
        *,
        remaining_tokens: int,
    ) -> Decision[QueryPlan]:
        return Decision(plan.model_copy(update={"sub_queries": []}))


class FakeCommerceWriter:
    async def ainvoke(self, graph_input: list[BaseMessage]) -> AIMessage:
        content = str(graph_input[0].content).split("Evidence:\n", 1)[-1]
        blocks = re.split(r"\n\n(?=\[E\d+\])", content)
        lines = []
        for block in blocks[:8]:
            label = re.match(r"\[(E\d+)\]", block)
            if label and "\n" in block:
                # Each line copies actual evidence verbatim and passes normal claim checks.
                text = block.split("\n", 1)[1].replace("\n", " ")[:1800]
                for part in re.split(r"(?<=[。!?])\s+(?=\S)", text):
                    lines.append(part + f" [{label.group(1)}]")
        return AIMessage(content="\n\n".join(lines))

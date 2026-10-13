"""Measured synthetic business checks, separate from QASPER/RAGBench results."""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from ai_agent_template.commerce.demo import manual
from ai_agent_template.commerce.importers import read_catalog
from ai_agent_template.commerce.orders import after_sales
from ai_agent_template.commerce.retrieval import CommerceScope, scope
from ai_agent_template.commerce.schemas import (
    AfterSalesRequest,
    ChatRequest,
    ListingRequest,
    Principal,
    Product,
    ProductVariant,
)
from ai_agent_template.commerce.service import CommerceService
from ai_agent_template.core.settings import Settings


async def evaluate(output: Path, live: bool = False) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    latencies = []
    metrics: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="commerce-eval-") as directory:
        settings = Settings().model_copy(
            update={
                "commerce_database_url": "sqlite:///" + str(Path(directory) / "commerce.sqlite3"),
                "rag_data_path": str(Path(directory) / "rag"),
                "qdrant_url": None,
                "commerce_mode": "demo",
                "commerce_llm_mode": "live" if live else "offline",
                "rag_top_k": 5,
            }
        )
        service = CommerceService(settings)
        # Explicit test fixture, never a runtime model fallback.
        from ai_agent_template.commerce.offline import FakeCommercePolicy, FakeCommerceWriter

        service._model, service._policy = (
            FakeCommerceWriter(),
            FakeCommercePolicy(service.repository),
        )
        user = Principal(store_id="demo-store", subject="evaluation", role="operator")
        try:
            products = {item.id: item for item in service.repository.list(Product, user.store_id)}
            variants = service.repository.list(ProductVariant, user.store_id)
            snapshot = {
                "products": [item.model_dump(mode="json") for item in products.values()],
                "variants": [item.model_dump(mode="json") for item in variants],
                "manuals": {item.sku: manual(item, products[item.product_id]) for item in variants},
            }
            output.joinpath("dataset_snapshot.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await service.seed_knowledge(user)
            hits = 0
            for variant in variants:
                token = scope.set(CommerceScope(user, "US", "en"))
                try:
                    started = time.perf_counter()
                    sources = service.retriever.search(
                        f"{variant.sku} specifications", mode="hybrid"
                    )
                    duration = (time.perf_counter() - started) * 1000
                finally:
                    scope.reset(token)
                ranked = list(dict.fromkeys(item.metadata.get("sku") for item in sources))[:5]
                hit = variant.sku in ranked
                hits += hit
                latencies.append(duration)
                results.append(
                    {
                        "check": "sku_recall_at_5",
                        "sku": variant.sku,
                        "ranked_skus": ranked,
                        "hit": hit,
                        "latency_ms": duration,
                    }
                )
            metrics["sku_recall_at_5"] = {
                "value": hits / len(variants),
                "evaluated": len(variants),
                "scope": "unfiltered hybrid retrieval over 20 synthetic SKU manuals",
            }
            multilingual = citations_valid = evaluated_citations = 0
            for language, question in (
                ("zh", "DEMO-SERUM-30 的容量是多少？"),
                ("en", "What is the capacity of DEMO-SERUM-30?"),
                ("es", "¿Cuál es la capacidad de DEMO-SERUM-30?"),
            ):
                started = time.perf_counter()
                result = await service.chat(
                    user, ChatRequest(message=question, language=language, thread_id=language)
                )
                duration = (time.perf_counter() - started) * 1000
                cited = result.get("citations", [])
                correct = (
                    bool(cited)
                    and "30" in result["answer"]
                    and all(
                        item["evidence"]["metadata"].get("sku") in {None, "DEMO-SERUM-30"}
                        for item in cited
                    )
                )
                multilingual += correct
                for item in cited:
                    evaluated_citations += 1
                    citations_valid += item["status"] in {"active", "legacy"}
                results.append(
                    {
                        "check": "multilingual_named_sku",
                        "language": language,
                        "question": question,
                        "correct": correct,
                        "latency_ms": duration,
                        "response": result,
                    }
                )
            metrics["multilingual_product_retrieval_accuracy"] = {
                "value": multilingual / 3,
                "evaluated": 3,
                "scope": "zh/en/es named-SKU queries; bilingual documents, not a pure cross-language semantic benchmark",
            }
            metrics["citation_validity"] = {
                "value": citations_valid / evaluated_citations if evaluated_citations else None,
                "evaluated": evaluated_citations,
            }
            draft = await service.listing(user, ListingRequest(sku="DEMO-SERUM-30", language="en"))
            metrics["listing_fact_consistency"] = {
                "value": int(draft.review_status != "blocked"),
                "evaluated": 1,
                "scope": "configured claim verification; offline is deterministic, live includes semantic judge",
            }
            results.append({"check": "listing", "draft": draft.model_dump()})
            order_checks = [
                ("DEMO-ORDER-1", "completed"),
                ("DEMO-ORDER-2", "in_transit"),
                ("DEMO-ORDER-3", "in_transit"),
            ]
            correct_orders = 0
            for order_id, expected in order_checks:
                actual = service.orders.query(user, order_id)
                correct_orders += actual["order"]["status"] == expected
                results.append({"check": "order", "expected_status": expected, "response": actual})
            metrics["order_status_accuracy"] = {
                "value": correct_orders / len(order_checks),
                "evaluated": len(order_checks),
            }
            rule_cases = [
                ("damaged", None, True),
                ("changed_mind", True, False),
                ("changed_mind", None, None),
                ("changed_mind", False, True),
            ]
            correct_rules = 0
            for reason, opened, expected_eligibility in rule_cases:
                response = after_sales(
                    service.repository,
                    user,
                    "DEMO-ORDER-1",
                    AfterSalesRequest(reason=reason, opened=opened),
                )
                correct_rules += response["eligible_for_review"] == expected_eligibility
                results.append(
                    {"check": "return_rule", "expected": expected_eligibility, "response": response}
                )
            metrics["after_sales_rule_accuracy"] = {
                "value": correct_rules / len(rule_cases),
                "evaluated": len(rule_cases),
            }
            data = (
                b"sku,product_id,name_en,capacity,price,stock\nIMPORT-1,P1,Imported,75 ml,13.25,9\n"
            )
            imported, _ = read_catalog(data, "catalog.csv", user.store_id)
            imported_variant = imported.variants[0]
            comparisons = [
                imported_variant.sku == "IMPORT-1",
                imported_variant.attributes.get("capacity") == "75 ml",
                str(imported_variant.price) == "13.25",
                imported_variant.stock == 9,
            ]
            metrics["product_attribute_extraction_accuracy"] = {
                "value": sum(comparisons) / len(comparisons),
                "evaluated": len(comparisons),
                "scope": "deterministic structured CSV importer",
            }
            metrics["retrieval_latency_ms"] = {
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "evaluated": len(latencies),
            }
            output.joinpath("config.json").write_text(
                json.dumps(
                    {
                        "mode": settings.commerce_llm_mode,
                        "synthetic": True,
                        "generator": settings.model_name
                        if live
                        else "FakeCommerceWriter/extractive",
                        "embedding": settings.commerce_embedding_model
                        if live
                        else "OFFLINE-hash-v1",
                        "limitations": [
                            "Small synthetic smoke set",
                            "Named-SKU multilingual checks are not general cross-language semantic accuracy",
                            "No real refund or publishing",
                            "No QASPER/RAGBench metrics reused",
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            service.close()
    output.joinpath("predictions.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.joinpath("metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/runs") / ("commerce-" + uuid4().hex[:12])
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(asyncio.run(evaluate(args.output, args.live)))

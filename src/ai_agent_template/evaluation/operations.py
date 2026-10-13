"""Reproducible operations evaluation. Ground truth is loaded only by this scorer."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from uuid import uuid4

from ai_agent_template.commerce.investigation import InvestigationEngine
from ai_agent_template.commerce.operations_models import InvestigationRequest
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.commerce.report_importer import import_report
from ai_agent_template.commerce.schemas import Principal
from ai_agent_template.commerce.service import CommerceService
from ai_agent_template.core.settings import Settings
from ai_agent_template.rag.repository import digest


async def evaluate(
    output: Path, fixtures: Path = Path("examples/commerce/operations")
) -> dict[str, Any]:
    truth = json.loads((fixtures / "ground-truth.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    latencies = []
    correct_fields = 0
    correct_metrics = 0
    correct_returns = 0
    valid_refs = 0
    all_refs = 0
    predicted_positive = set()
    with TemporaryDirectory(prefix="ops-eval-") as directory:
        settings = Settings(
            _env_file=None,
            commerce_database_url=f"sqlite:///{directory}/business.sqlite3",
            rag_data_path=f"{directory}/rag",
            commerce_seed_demo=False,
            commerce_llm_mode="offline",
            commerce_principals={},
            openai_api_key=None,
            groq_api_key=None,
            commerce_mode="demo",
        )
        commerce = CommerceService(settings)
        repository = OperationsRepository(commerce.repository)
        for kind in (
            "supplier",
            "listing",
            "listing_issue",
            "ads",
            "search_terms",
            "sales",
            "returns",
            "revision",
            "quality",
        ):
            import_report(
                repository,
                kind,
                "evaluation",
                "US",
                (fixtures / f"{kind}.json").read_bytes(),
                f"{kind}.json",
                synthetic=True,
            )
        engine = InvestigationEngine(commerce, repository)
        principal = Principal(
            store_id="evaluation", subject="offline-evaluator", role="operator", marketplaces=["US"]
        )
        for index in range(101, 121):
            sku = f"SKU-A{index}"
            request = InvestigationRequest(
                seller_id="evaluation",
                marketplace_id="US",
                sku=sku,
                asin=f"DEMO{index:06d}",
                start="2026-09-01",
                end="2026-09-30",
                comparison_start="2026-08-01",
                comparison_end="2026-08-31",
                as_of="2026-10-13",
                question="显式模块确定性验收",
                modules=["listing", "returns", "ads"],
            )
            started = perf_counter()
            report = await engine.run(principal, request)
            latencies.append((perf_counter() - started) * 1000)
            predictions.append(report.model_dump(mode="json"))
            fields = {item["field"] for item in report.results["listing"]["field_differences"]}
            correct_fields += fields == set(truth["listing_mismatch_fields"].get(sku, []))
            group = report.results["ads"]["groups"][0]
            totals, metrics = group["totals"], group["metrics"]
            expected = dict(
                CTR=totals["clicks"] / totals["impressions"],
                CPC=totals["spend"] / totals["clicks"],
                CVR=totals["attributed_orders"] / totals["clicks"],
                ACOS=totals["spend"] / totals["attributed_sales"],
                ROAS=totals["attributed_sales"] / totals["spend"],
            )
            correct_metrics += sum(
                abs(metrics[key] - value) < 1e-9 for key, value in expected.items()
            )
            rate = truth["returns_rates"].get(sku, {}).get("2026-09", 0.02)
            correct_returns += report.results["returns"]["return_rate"] == rate
            if any(
                item["type"] == "high_spend_low_sales"
                for item in report.results["ads"]["anomalies"]
            ):
                predicted_positive.add(sku)
            known = {item.evidence_id for item in report.evidence}
            for finding in report.findings:
                all_refs += len(finding.evidence_ids)
                valid_refs += sum(eid in known for eid in finding.evidence_ids)
        commerce.close()
    actual = set(truth["ads_high_spend_skus"])
    matched = len(actual & predicted_positive)
    metrics = dict(
        cases=len(predictions),
        listing_exact_field_accuracy=correct_fields / len(predictions),
        ads_numeric_correctness=correct_metrics / (5 * len(predictions)),
        ads_anomaly_precision=matched / len(predicted_positive) if predicted_positive else None,
        ads_anomaly_recall=matched / len(actual) if actual else None,
        returns_numeric_correctness=correct_returns / len(predictions),
        citation_validity=valid_refs / all_refs if all_refs else None,
        task_completion=sum(not item["failed_tools"] for item in predictions) / len(predictions),
        mean_latency_ms=sum(latencies) / len(latencies),
        source_sha256={path.name: digest(path.read_bytes()) for path in fixtures.glob("*.json")},
        mode="deterministic_offline_synthetic_only",
        exclusions=[
            "不是通用 LLM 准确率",
            "未评测原因分类器 F1",
            "旧 QASPER/RAGBench/RAGAS 独立保留",
        ],
    )
    (output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    destination = Path("experiments/runs") / ("operations-" + uuid4().hex[:12])
    print(json.dumps(asyncio.run(evaluate(destination)), ensure_ascii=False, indent=2))
    print("Artifact:", destination)

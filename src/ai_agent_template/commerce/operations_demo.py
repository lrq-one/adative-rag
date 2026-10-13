"""Reproducible synthetic reports; evaluator-only labels never enter planner input."""

import json
from pathlib import Path
from typing import Any

from ai_agent_template.commerce.operations_models import SupplierDocument
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.commerce.report_importer import import_report


def demo_reports() -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {
        kind: []
        for kind in (
            "supplier",
            "listing",
            "listing_issue",
            "revision",
            "sales",
            "returns",
            "quality",
            "ads",
            "search_terms",
            "campaign",
        )
    }
    for index in range(101, 121):
        sku = f"SKU-A{index}"
        anomalous = index == 102  # Fixture generation only; diagnostic code never branches on SKU.
        for period, day in ((1, "2026-08-01"), (2, "2026-09-01")):
            base = dict(
                sku=sku,
                asin=f"DEMO{index:06d}",
                product_id=f"product-{index}",
                product_version=str(period),
                effective_at=day,
                report_period=day[:7],
                version=period,
                synthetic=True,
            )
            attributes = dict(
                capacity="30ml",
                color="clear",
                package_quantity="1",
                packaging="carton-v2" if period == 2 and anomalous else "carton-v1",
                category="skincare" if index < 111 else "beauty-tool",
            )
            reports["supplier"].append(
                base
                | dict(
                    attributes=attributes,
                    description="Synthetic product: 30ml skincare serum / 护肤精华；无防晒或医疗功效声明。",
                )
            )
            listing_attributes = attributes | (
                {"capacity": "60ml"} if period == 2 and anomalous else {}
            )
            reports["listing"].append(
                base | dict(title=f"Demo product {index}", attributes=listing_attributes)
            )
            reports["sales"].append(
                base | dict(sold_date=day[:8] + "10", units=1000, cohort_id=f"{sku}-{period}")
            )
            reports["returns"].append(
                base
                | dict(
                    return_date=day[:8] + "20",
                    sold_date=day[:8] + "10",
                    units=80 if period == 2 and anomalous else 20,
                    reason="package_damaged" if period == 2 and anomalous else "changed_mind",
                    cohort_id=f"{sku}-{period}",
                )
            )
            reports["quality"].append(
                base
                | dict(
                    inspection_date=day[:8] + "02",
                    batch_id=f"batch-{sku}-{period}",
                    result="Synthetic spot inspection: no causal conclusion",
                )
            )
            reports["campaign"].append(
                base | dict(campaign_id=f"campaign-{sku}", name=f"Demo {sku}", ad_type="SP")
            )
            metrics = dict(
                report_date=day[:8] + "15",
                campaign_id=f"campaign-{sku}",
                ad_group_id="group-1",
                currency="USD",
                attribution_window=7,
                impressions=10000,
                clicks=200 if period == 2 and anomalous else 100,
                spend=300 if period == 2 and anomalous else 100,
                attributed_sales=100 if period == 2 and anomalous else 1000,
                attributed_orders=5 if period == 2 and anomalous else 50,
            )
            reports["ads"].append(base | metrics)
            reports["search_terms"].append(
                base
                | metrics
                | dict(
                    search_term="sunscreen spf50"
                    if period == 2 and anomalous
                    else "skincare serum",
                    keyword="serum",
                    match_type="BROAD",
                )
            )
            if period == 2 and anomalous:
                reports["revision"].append(
                    base
                    | dict(
                        change_type="packaging",
                        description="Synthetic packaging revision: protective insert removed; requires batch validation",
                        previous_version="1",
                    )
                )
                reports["listing_issue"].append(
                    base
                    | dict(
                        field="capacity",
                        message="Synthetic internal import: check capacity",
                        severity="high",
                    )
                )
    return reports


def seed_operations(repo: OperationsRepository, seller: str = "demo-store") -> dict[str, Any]:
    if repo.list(SupplierDocument, seller, "US"):
        return {"seeded": False}
    batches = [
        import_report(
            repo, kind, seller, "US", json.dumps(rows).encode(), f"demo-{kind}.json", synthetic=True
        )
        for kind, rows in demo_reports().items()
        if rows
    ]
    return dict(seeded=True, batches=batches, synthetic=True)


def export_demo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for kind, rows in demo_reports().items():
        (path / f"{kind}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (path / "supplier-SKU-A102.md").write_text(
        "# 模拟供应商资料 / Synthetic supplier manual\nSKU-A102，30ml 护肤精华，包装版本2。\n包装变更于2026-09-01，必须按批次质检，不能认定变更导致退货。\nThis product has no SPF or sunscreen claim.\n",
        encoding="utf-8",
    )
    (path / "ground-truth.json").write_text(
        json.dumps(
            {
                "fixture_only": True,
                "listing_mismatch_fields": {"SKU-A102": ["capacity"]},
                "ads_high_spend_skus": ["SKU-A102"],
                "returns_rates": {"SKU-A102": {"2026-08": 0.02, "2026-09": 0.08}},
                "causal_label": "not_established",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    export_demo(Path("examples/commerce/operations"))

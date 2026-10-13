"""Deterministic diagnostics; measured facts, correlations and hypotheses stay separate."""

from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, TypeVar

from ai_agent_template.commerce.operations_models import (
    AdsPerformance,
    AdsSearchTerm,
    AnalysisRequest,
    BusinessEvidence,
    ListingIssue,
    ListingSnapshot,
    OperationsRecord,
    ProductRevision,
    QualityRecord,
    ReturnRecord,
    RuleRecord,
    SalesRecord,
    SupplierDocument,
)
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.rag.repository import fingerprint

T = TypeVar("T", bound=OperationsRecord)


def evidence(
    request: AnalysisRequest,
    records: list[OperationsRecord],
    definition: str,
    calculation: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(
        source_type="structured_report",
        seller_id=request.seller_id,
        marketplace=request.marketplace_id,
        sku=request.sku,
        import_batch_id=sorted({item.import_batch_id for item in records}),
        report_name=sorted({item.source_id for item in records}),
        row_ids=[item.id for item in records],
        time_range={"start": str(request.start), "end": str(request.end)},
        metric_definition=definition,
        filters=request.model_dump(mode="json"),
        calculation=calculation,
        source_version=",".join(sorted({item.product_version for item in records})),
    )
    return BusinessEvidence(evidence_id="B-" + fingerprint(payload)[:24], **payload).model_dump(
        mode="json"
    )


def ratio(numerator: int | Decimal | None, denominator: int | Decimal | None) -> float | None:
    return (
        float(numerator / denominator)
        if numerator is not None and denominator is not None and denominator > 0
        else None
    )


class OperationsAnalytics:
    def __init__(self, repository: OperationsRepository) -> None:
        self.repository = repository

    def scoped(self, cls: type[T], request: AnalysisRequest) -> list[T]:
        return [
            item
            for item in self.repository.list(
                cls, request.seller_id, request.marketplace_id, request.sku
            )
            if not request.asin or item.asin == request.asin
        ]

    def validate_product(self, request: AnalysisRequest) -> None:
        if not self.scoped(SupplierDocument, request):
            raise LookupError("No product mapping/supplier data in selected seller and marketplace")
        identities = {
            (item.product_id, item.asin) for item in self.scoped(SupplierDocument, request)
        }
        if len(identities) > 1 and not request.asin:
            raise ValueError("Multiple SKU/ASIN mappings: select ASIN explicitly")

    def latest(self, cls: type[T], request: AnalysisRequest) -> list[T]:
        candidates = [
            item
            for item in self.scoped(cls, request)
            if item.effective_at <= request.end
            and (item.expires_at is None or item.expires_at > request.end)
        ]
        if not candidates:
            return []
        newest = max((item.effective_at, item.version) for item in candidates)
        return [item for item in candidates if (item.effective_at, item.version) == newest]

    def listing(self, request: AnalysisRequest) -> dict[str, Any]:
        self.validate_product(request)
        suppliers = self.latest(SupplierDocument, request)
        snapshots = self.latest(ListingSnapshot, request)
        issues: list[dict[str, Any]] = []
        refs: list[dict[str, Any]] = []
        missing: list[str] = []
        if not snapshots:
            missing.append("没有所选日期有效的 Listing 快照")
        if not suppliers:
            missing.append("没有所选日期有效的供应商规格")
        if len({fingerprint([item.attributes, item.product_version]) for item in suppliers}) > 1:
            missing.append("同一有效时间存在冲突的供应商版本，需人工选择权威版本")
        if len({fingerprint([item.attributes, item.product_version]) for item in snapshots}) > 1:
            missing.append("同一有效时间存在冲突的 Listing 快照")
        if not missing:
            supplier, snapshot = suppliers[0], snapshots[0]
            expected = {**supplier.attributes, "product_version": supplier.product_version}
            actual = {**snapshot.attributes, "product_version": snapshot.product_version}
            rules = [
                item
                for item in self.scoped(RuleRecord, request)
                if item.effective_at <= request.end
                and (not item.expires_at or item.expires_at > request.end)
            ]
            required = set(expected) | {field for rule in rules for field in rule.required_fields}
            for field in sorted(required):
                left, right = expected.get(field), actual.get(field)
                if right is None or (
                    left is not None
                    and str(left).strip().casefold() != str(right).strip().casefold()
                ):
                    ref = evidence(
                        request,
                        [supplier, snapshot, *rules],
                        "字段逐项对比：最新有效供应商规格 vs Listing 快照",
                        {
                            "field": field,
                            "expected_value": left,
                            "actual_value": right,
                            "supplier_version": supplier.product_version,
                            "listing_version": snapshot.product_version,
                        },
                    )
                    refs.append(ref)
                    issues.append(
                        dict(
                            issue_type="missing_attribute"
                            if right is None
                            else "attribute_mismatch",
                            field=field,
                            expected_value=left,
                            actual_value=right,
                            source_reference=ref["evidence_id"],
                            severity="high"
                            if field
                            in {"capacity", "dimensions", "package_quantity", "product_version"}
                            else "medium",
                            reason="Listing 与来源规格不一致，需要人工复核",
                        )
                    )
        for item in self.scoped(ListingIssue, request):
            if item.effective_at <= request.end and (
                not item.expires_at or item.expires_at > request.end
            ):
                ref = evidence(
                    request,
                    [item],
                    "导入的 Listing 问题报告（未独立确认平台结论）",
                    {"field": item.field, "message": item.message},
                )
                refs.append(ref)
                issues.append(
                    dict(
                        issue_type="imported_issue",
                        field=item.field,
                        expected_value=None,
                        actual_value=None,
                        source_reference=ref["evidence_id"],
                        severity=item.severity,
                        reason=item.message,
                    )
                )
        return dict(
            sku=request.sku,
            marketplace=request.marketplace_id,
            issues=issues,
            field_differences=[item for item in issues if item["issue_type"] != "imported_issue"],
            evidence=refs,
            suggested_changes=[
                dict(
                    field=item["field"],
                    draft_value=item["expected_value"],
                    evidence_id=item["source_reference"],
                    priority=item["severity"],
                )
                for item in issues
                if item["expected_value"] is not None
            ],
            requires_review=True,
            missing_evidence=missing,
            audit_scope="内部一致性检查；不代表 Amazon 合规审核",
            actions_executed=[],
        )

    def _return_window(self, request: AnalysisRequest) -> dict[str, Any]:
        sales = [
            item
            for item in self.scoped(SalesRecord, request)
            if request.start <= item.sold_date <= request.end
        ]
        returns = self.scoped(ReturnRecord, request)
        as_of = request.as_of or date.today()
        gaps: list[OperationsRecord] = []
        if request.return_basis == "sales_cohort":
            gaps = [item for item in returns if item.sold_date is None or not item.cohort_id]
            returns = [
                item
                for item in returns
                if item.sold_date is not None
                and request.start <= item.sold_date <= request.end
                and item.return_date <= as_of
                and item.cohort_id in {sale.cohort_id for sale in sales if sale.cohort_id}
            ]
            if any(not sale.cohort_id for sale in sales):
                gaps.extend(sales)
        else:
            returns = [item for item in returns if request.start <= item.return_date <= request.end]
        units = sum(item.units for item in returns)
        sold = sum(item.units for item in sales) if sales else None
        reasons: Counter[str] = Counter()
        for item in returns:
            reasons[item.reason] += item.units
        rate = ratio(units, sold) if not gaps and returns else None
        definition = "退货件数/销售件数；event_period 为各自事件日期窗口（非订单 cohort，存在滞后）；sales_cohort 为 sold_date 同窗口且 cohort_id 匹配、截至 as_of 已观测退货"
        ref = evidence(
            request,
            [*sales, *returns],
            definition,
            {
                "returned_units": units,
                "sold_units": sold,
                "return_rate": rate,
                "reasons": dict(reasons),
                "as_of": str(as_of),
                "basis": request.return_basis,
                "cohort_missing_rows": len(gaps),
            },
        )
        return dict(
            returned_units=units,
            sold_units=sold,
            return_rate=rate,
            reasons=dict(reasons),
            evidence=[ref],
            missing_evidence=["销售分母缺失/为零或 cohort 归属不完整，不能计算可比退货率"]
            if rate is None
            else [],
        )

    def returns(self, request: AnalysisRequest) -> dict[str, Any]:
        self.validate_product(request)
        result = self._return_window(request)
        comparison = (
            self._return_window(
                request.model_copy(
                    update={"start": request.comparison_start, "end": request.comparison_end}
                )
            )
            if request.comparison_start
            else None
        )
        revisions = [
            item
            for item in self.scoped(ProductRevision, request)
            if request.start <= item.effective_at <= request.end
        ]
        quality = [
            item
            for item in self.scoped(QualityRecord, request)
            if request.start <= item.inspection_date <= request.end
        ]
        ref = evidence(
            request,
            [*revisions, *quality],
            "产品/包装变更和质检记录的时间线；不能证明因果",
            {
                "revisions": [item.model_dump(mode="json") for item in revisions],
                "quality": [item.model_dump(mode="json") for item in quality],
            },
        )
        facts = [
            dict(
                text="所选窗口销售及退货件数已汇总",
                evidence_ids=[result["evidence"][0]["evidence_id"]],
            )
        ]
        if revisions:
            facts.append(
                dict(text="窗口内存在商品或包装版本变更记录", evidence_ids=[ref["evidence_id"]])
            )
        correlations = []
        if (
            comparison
            and result["return_rate"] is not None
            and comparison["return_rate"] is not None
            and result["return_rate"] > comparison["return_rate"]
        ):
            correlations.append(
                dict(
                    text="本窗口事件期退货率高于对照窗口；若存在版本变更，仅构成时间上的共现",
                    evidence_ids=[
                        result["evidence"][0]["evidence_id"],
                        comparison["evidence"][0]["evidence_id"],
                        ref["evidence_id"],
                    ],
                )
            )
        return {
            **result,
            "comparison": comparison,
            "timeline": [item.model_dump(mode="json") for item in revisions],
            "quality_records": [item.model_dump(mode="json") for item in quality],
            "evidence": [*result["evidence"], *(comparison["evidence"] if comparison else []), ref],
            "confirmed_facts": facts,
            "observed_correlations": correlations,
            "hypotheses": [
                dict(
                    text="版本变更可能影响质量或预期，需要按批次对照验证",
                    evidence_ids=[ref["evidence_id"]],
                )
            ]
            if revisions and result["returned_units"]
            else [],
            "missing_evidence": result["missing_evidence"]
            + ["事件期指标存在退货滞后；确认因果需要订单 cohort、批次及物流/质检对照"],
            "recommended_checks": ["对照退货对应销售批次、包装版本、运输条件和质检样本"],
            "actions_executed": [],
        }

    def _ads_window(self, request: AnalysisRequest, *, terms_only: bool = False) -> dict[str, Any]:
        records: list[AdsPerformance] = (
            list(self.scoped(AdsSearchTerm, request))
            if terms_only or request.search_term
            else list(self.scoped(AdsPerformance, request))
        )
        # Search-term reports and campaign reports overlap. Never sum both granularities.
        if not records:
            records = list(self.scoped(AdsSearchTerm, request))
        records = [
            item
            for item in records
            if request.start <= item.report_date <= request.end
            and (not request.campaign_id or item.campaign_id == request.campaign_id)
            and (not request.ad_group_id or item.ad_group_id == request.ad_group_id)
            and (not request.search_term or item.search_term == request.search_term)
        ]
        groups: dict[tuple[str, str, int, str, str, str], list[AdsPerformance]] = defaultdict(list)
        for item in records:
            groups[
                (
                    item.currency,
                    item.ad_type,
                    item.attribution_window,
                    item.campaign_id,
                    item.ad_group_id,
                    item.search_term,
                )
            ].append(item)
        output = []
        refs = []
        for key, items in sorted(groups.items()):
            totals: dict[str, Any] = {}
            for field in (
                "impressions",
                "clicks",
                "spend",
                "attributed_sales",
                "attributed_orders",
            ):
                values = [getattr(item, field) for item in items]
                totals[field] = sum(values) if all(value is not None for value in values) else None
            metrics = {
                name: ratio(totals[num], totals[den])
                for name, num, den in (
                    ("CTR", "clicks", "impressions"),
                    ("CPC", "spend", "clicks"),
                    ("CVR", "attributed_orders", "clicks"),
                    ("ACOS", "spend", "attributed_sales"),
                    ("ROAS", "attributed_sales", "spend"),
                )
            }
            as_of = request.as_of or date.today()
            immature = any(
                item.report_date + timedelta(days=item.attribution_window) > as_of for item in items
            )
            serial_totals = {
                field: float(value) if isinstance(value, Decimal) else value
                for field, value in totals.items()
            }
            ref = evidence(
                request,
                list(items),
                "CTR=clicks/impressions; CPC=spend/clicks; CVR=attributed_orders/clicks; ACOS=spend/attributed_sales; ROAS=attributed_sales/spend. 零/缺失分母=>null。分币种/类型/归因窗口/campaign/adgroup/term。",
                {
                    "totals": serial_totals,
                    "metrics": metrics,
                    "group": list(key),
                    "attribution_immature": immature,
                },
            )
            refs.append(ref)
            output.append(
                dict(
                    currency=key[0],
                    ad_type=key[1],
                    attribution_window=key[2],
                    campaign_id=key[3],
                    ad_group_id=key[4],
                    search_term=key[5],
                    totals=serial_totals,
                    metrics=metrics,
                    attribution_immature=immature,
                    insufficient_sample=totals["clicks"] is None or totals["clicks"] < 20,
                    evidence_id=ref["evidence_id"],
                )
            )
        return dict(groups=output, evidence=refs)

    def ads(self, request: AnalysisRequest) -> dict[str, Any]:
        self.validate_product(request)
        current = self._ads_window(request)
        terms = self._ads_window(request, terms_only=True)
        comparison = (
            self._ads_window(
                request.model_copy(
                    update={"start": request.comparison_start, "end": request.comparison_end}
                )
            )
            if request.comparison_start
            else dict(groups=[], evidence=[])
        )

        def key(group: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(
                group[field]
                for field in (
                    "currency",
                    "ad_type",
                    "attribution_window",
                    "campaign_id",
                    "ad_group_id",
                    "search_term",
                )
            )

        previous = {key(group): group for group in comparison["groups"]}
        anomalies = []
        uncertainties = []
        for group in current["groups"]:
            before = previous.get(key(group))
            refs = [group["evidence_id"]]
            if before:
                refs.append(before["evidence_id"])
            if group["attribution_immature"] or group["insufficient_sample"]:
                uncertainties.append(
                    dict(
                        group=list(key(group)),
                        reason="归因尚未成熟或 clicks<20，建议人工复核",
                        evidence_ids=refs,
                    )
                )
            if (
                group["totals"]["spend"] is not None
                and group["totals"]["spend"] >= 50
                and group["totals"]["attributed_sales"] is not None
                and group["totals"]["attributed_sales"] < group["totals"]["spend"]
            ):
                anomalies.append(
                    dict(
                        type="high_spend_low_sales",
                        group=list(key(group)),
                        evidence_ids=refs,
                        reason="spend≥50 报表币种单位且 attributed_sales<spend；阈值是内部规则",
                    )
                )
            if before and not group["attribution_immature"] and not before["attribution_immature"]:
                for metric in ("ACOS", "CPC"):
                    old, new = before["metrics"][metric], group["metrics"][metric]
                    if old is not None and new is not None and old > 0 and new > old * 1.25:
                        anomalies.append(
                            dict(
                                type=metric.lower() + "_rise",
                                group=list(key(group)),
                                evidence_ids=refs,
                                before=old,
                                after=new,
                                reason="比同口径对照窗口上升超过25%（内部阈值）",
                            )
                        )
                equal_duration = (
                    (request.end - request.start).days
                    == (request.comparison_end - request.comparison_start).days
                    if request.comparison_end and request.comparison_start
                    else False
                )
                if (
                    equal_duration
                    and group["totals"]["clicks"] is not None
                    and before["totals"]["clicks"] is not None
                    and group["totals"]["clicks"] > before["totals"]["clicks"]
                    and group["metrics"]["CVR"] is not None
                    and before["metrics"]["CVR"] is not None
                    and group["metrics"]["CVR"] < before["metrics"]["CVR"]
                ):
                    anomalies.append(
                        dict(
                            type="clicks_up_cvr_down",
                            group=list(key(group)),
                            evidence_ids=refs,
                            reason="等长窗口点击增加且 CVR 下降",
                        )
                    )
        return dict(
            groups=current["groups"],
            comparison_groups=comparison["groups"],
            anomalies=anomalies,
            metric_evidence=[*current["evidence"], *comparison["evidence"], *terms["evidence"]],
            evidence=[*current["evidence"], *comparison["evidence"], *terms["evidence"]],
            search_term_metrics=terms["groups"],
            search_term_findings=[],
            product_evidence=[],
            candidate_actions=[
                dict(
                    action="人工检查高消耗搜索词与商品用途/规格的匹配和归因成熟度；暂不自动否词",
                    evidence_ids=item["evidence_ids"],
                    requires_review=True,
                )
                for item in anomalies
            ],
            uncertainties=uncertainties,
            missing_evidence=["无所选窗口广告报告"] if not current["groups"] else [],
            actions_executed=[],
        )

import json

import pytest

from ai_agent_template.commerce.analytics import OperationsAnalytics
from ai_agent_template.commerce.operations_models import AnalysisRequest, SupplierDocument
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.commerce.report_importer import import_report
from ai_agent_template.commerce.repository import CommerceRepository


@pytest.fixture
def repo():
    repository = CommerceRepository("sqlite:///:memory:")
    yield OperationsRepository(repository)
    repository.engine.dispose()


def supplier(**updates):
    return (
        dict(
            sku="same",
            product_id="product",
            effective_at="2026-09-01",
            attributes={"capacity": "30ml"},
        )
        | updates
    )


def load(repo, kind, rows, market="US", seller="seller"):
    return import_report(repo, kind, seller, market, json.dumps(rows).encode(), kind + ".json")


def request(**updates):
    values = dict(
        seller_id="seller",
        marketplace_id="US",
        sku="same",
        start="2026-09-01",
        end="2026-09-30",
        as_of="2026-10-13",
    )
    return AnalysisRequest(**(values | updates))


def test_import_scope_and_idempotency(repo):
    batch = load(repo, "supplier", [supplier(), supplier()])
    assert batch["rows"] == 1 and batch["duplicate_rows"] == 1
    assert load(repo, "supplier", [supplier(), supplier()])["idempotent"]
    load(repo, "supplier", [supplier(attributes={"capacity": "60ml"})], market="DE")
    assert repo.list(SupplierDocument, "seller", "US")[0].attributes["capacity"] == "30ml"
    assert repo.list(SupplierDocument, "seller", "DE")[0].attributes["capacity"] == "60ml"
    assert repo.list(SupplierDocument, "other", "US") == []


def test_invalid_report_is_atomic(repo):
    with pytest.raises(ValueError, match="Row 3"):
        load(repo, "supplier", [supplier(), supplier(effective_at="invalid")])
    assert repo.list(SupplierDocument, "seller", "US") == []
    with pytest.raises(PermissionError):
        load(repo, "supplier", [supplier(seller_id="other")])


def test_window_validation():
    with pytest.raises(ValueError):
        request(comparison_start="2026-09-01", comparison_end="2026-09-30")


def test_listing_difference_then_normal(repo):
    load(repo, "supplier", [supplier()])
    listing = supplier(attributes={"capacity": "60ml"})
    load(repo, "listing", [listing])
    analytics = OperationsAnalytics(repo)
    result = analytics.listing(request())
    assert result["issues"][0]["field"] == "capacity"
    assert result["evidence"][0]["calculation"]["expected_value"] == "30ml"
    assert "合规" in result["audit_scope"]
    load(repo, "listing", [supplier(effective_at="2026-09-02")])
    assert analytics.listing(request())["issues"] == []
    with pytest.raises(LookupError):
        analytics.listing(request(sku="missing"))


def test_conflicting_supplier_versions(repo):
    load(repo, "supplier", [supplier(), supplier(attributes={"capacity": "60ml"})])
    load(repo, "listing", [supplier()])
    result = OperationsAnalytics(repo).listing(request())
    assert result["missing_evidence"] and not result["suggested_changes"]


def test_returns_units_and_missing_denominator(repo):
    load(repo, "supplier", [supplier()])
    load(
        repo,
        "returns",
        [
            supplier(attributes=None)
            | {"return_date": "2026-09-10", "units": 3, "reason": "damaged"}
        ],
    )
    analytics = OperationsAnalytics(repo)
    assert analytics.returns(request())["return_rate"] is None
    load(repo, "sales", [supplier(attributes=None) | {"sold_date": "2026-09-03", "units": 100}])
    result = analytics.returns(request())
    assert result["return_rate"] == 0.03 and result["reasons"] == {"damaged": 3}
    assert not result["hypotheses"]
    load(
        repo,
        "revision",
        [
            supplier(attributes=None)
            | {"change_type": "packaging", "description": "new carton", "previous_version": "0"}
        ],
    )
    result = analytics.returns(request())
    assert result["hypotheses"] and "对照" in result["recommended_checks"][0]
    assert result["evidence"][0]["calculation"]["sold_units"] == 100


def ads_row(**updates):
    return (
        supplier(attributes=None)
        | dict(
            report_date="2026-09-10",
            campaign_id="c",
            ad_group_id="g",
            currency="USD",
            attribution_window=7,
            impressions=1000,
            clicks=100,
            spend=100,
            attributed_sales=200,
            attributed_orders=10,
        )
        | updates
    )


@pytest.mark.parametrize(
    "changes,metric",
    [
        ({"attributed_sales": 0}, "ACOS"),
        ({"clicks": 0}, "CPC"),
        ({"impressions": 0}, "CTR"),
        ({"spend": 0}, "ROAS"),
        ({"clicks": None}, "CVR"),
    ],
)
def test_ads_zero_and_missing(repo, changes, metric):
    load(repo, "supplier", [supplier()])
    load(repo, "ads", [ads_row(**changes)])
    result = OperationsAnalytics(repo).ads(request())
    assert result["groups"][0]["metrics"][metric] is None
    assert not result["actions_executed"]


def test_ads_currency_window_and_type_not_mixed(repo):
    load(repo, "supplier", [supplier()])
    load(
        repo,
        "ads",
        [ads_row(), ads_row(currency="EUR"), ads_row(attribution_window=14), ads_row(ad_type="SB")],
    )
    result = OperationsAnalytics(repo).ads(request())
    assert len(result["groups"]) == 4
    assert all(group["totals"]["spend"] == 100 for group in result["groups"])


@pytest.fixture
def system(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from ai_agent_template.api.app import create_app
    from ai_agent_template.api.routes import commerce, operations
    from ai_agent_template.commerce.service import CommerceService
    from ai_agent_template.core.settings import Settings

    settings = Settings(
        _env_file=None,
        commerce_database_url=f"sqlite:///{tmp_path / 'business.sqlite3'}",
        rag_data_path=str(tmp_path / "rag"),
        commerce_seed_demo=False,
        commerce_llm_mode="offline",
        commerce_mode="demo",
        commerce_principals={},
        openai_api_key=None,
        groq_api_key=None,
        rate_limit_enabled=False,
        auth_api_keys=[],
    )
    service = CommerceService(settings)
    engine = operations.get_operations(service)
    monkeypatch.setattr(commerce, "get_settings", lambda: settings)
    client = TestClient(create_app())
    client.app.dependency_overrides[commerce.get_commerce_service] = lambda: service
    client.app.dependency_overrides[operations.get_operations] = lambda: engine
    client.headers["Authorization"] = "Bearer demo-operator"
    yield client, engine, settings
    service.close()


def body(**updates):
    return (
        request(seller_id="demo-store").model_dump(mode="json")
        | dict(
            question="商品规格和退货广告调查", modules=["listing", "returns", "ads", "knowledge"]
        )
        | updates
    )


def test_http_import_upload_investigate_and_original(system):
    client, engine, _ = system
    attrs = {"capacity": "30ml"}
    upload = client.post(
        "/v1/operations/documents",
        data=dict(
            seller_id="demo-store",
            marketplace_id="US",
            sku="same",
            asin="demo",
            product_id="product",
            effective_at="2026-09-01",
            product_version="1",
            attributes=json.dumps(attrs),
        ),
        files={
            "file": ("supplier.md", b"# Supplier\nCapacity 30ml. Packaging requires inspection.")
        },
    )
    assert upload.status_code == 200, upload.text
    payload = supplier(asin="demo", attributes={"capacity": "60ml"})
    for kind, records in [
        ("listing", [payload]),
        ("ads", [ads_row(asin="demo")]),
        (
            "returns",
            [
                supplier(asin="demo", attributes=None)
                | dict(return_date="2026-09-10", units=3, reason="damage")
            ],
        ),
        (
            "sales",
            [supplier(asin="demo", attributes=None) | dict(sold_date="2026-09-03", units=100)],
        ),
    ]:
        response = client.post(
            f"/v1/operations/imports/{kind}",
            data=dict(seller_id="demo-store", marketplace_id="US"),
            files={"file": (kind + ".json", json.dumps(records).encode())},
        )
        assert response.status_code == 200, response.text
    response = client.post(
        "/v1/operations/investigations", json=body(asin="demo", question="Capacity 30ml 包装质量")
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["results"]["returns"]["return_rate"] == 0.03
    assert report["results"]["ads"]["groups"][0]["metrics"]["ACOS"] == 0.5
    assert report["results"]["listing"]["issues"] and report["document_evidence"]
    assert report["verification"]["metrics_recomputed"]
    assert "60ml" in report["answer"] and "30ml" in report["answer"]
    assert "3%" in report["answer"] and "50%" in report["answer"]
    assert "建议" in report["answer"] and report["answer_mode"] == "verified_data"
    assert {
        eid for paragraph in report["answer_paragraphs"] for eid in paragraph["evidence_ids"]
    } <= {ref["evidence_id"] for ref in report["evidence"]}
    stored = client.get(f"/v1/operations/investigations/{report['investigation']['id']}")
    assert stored.status_code == 200 and stored.json()["answer"] == report["answer"]
    first = report["evidence"][0]
    traced = client.get(
        "/v1/operations/rows", params=dict(marketplace_id="US", row_ids=",".join(first["row_ids"]))
    )
    assert traced.status_code == 200 and traced.json()
    assert set(ref for finding in report["findings"] for ref in finding["evidence_ids"]) <= {
        ref["evidence_id"] for ref in report["evidence"]
    }
    stream = client.post("/v1/operations/investigations/stream", json=body(asin="demo"))
    assert "event: tool" in stream.text and "event: report" in stream.text
    assert '"answer_paragraphs"' in stream.text
    doc = upload.json()["document"]
    original = client.get(
        f"/v1/operations/documents/{doc['document_id']}/{doc['document_version']}/file",
        params=dict(marketplace_id="US", sku="same", asin="demo"),
    )
    assert original.status_code == 200 and b"Capacity 30ml" in original.content
    denied = client.get(
        f"/v1/operations/documents/{doc['document_id']}/{doc['document_version']}/file",
        params=dict(marketplace_id="DE", sku="same", asin="demo"),
    )
    assert denied.status_code == 404


def test_graph_single_module_and_permissions(system):
    client, engine, _ = system
    load(engine.repository, "supplier", [supplier()], seller="demo-store")
    load(engine.repository, "ads", [ads_row()], seller="demo-store")
    response = client.post("/v1/operations/investigations", json=body(modules=["ads"]))
    assert response.status_code == 200, response.text
    assert set(response.json()["results"]) == {"ads"}
    assert "ACOS" in response.json()["answer"] and "退货" not in response.json()["answer"]
    assert (
        client.post("/v1/operations/investigations", json=body(seller_id="other")).status_code
        == 403
    )
    assert (
        client.post("/v1/operations/investigations", json=body(marketplace_id="JP")).status_code
        == 403
    )
    assert client.post("/v1/operations/investigations", json=body(use_llm=True)).status_code == 503
    assert client.post("/v1/operations/investigations", json=body(modules=[])).status_code == 400


def test_illegal_llm_plan_is_rejected(system, monkeypatch):
    from ai_agent_template.agent.decisions import Decision, LLMQueryPolicy

    client, engine, _ = system
    load(engine.repository, "supplier", [supplier()], seller="demo-store")
    monkeypatch.setattr(engine, "model", lambda: object())

    async def invalid(*args, **kwargs):
        return Decision(None, error="invalid_structured_output")

    monkeypatch.setattr(LLMQueryPolicy, "_call", invalid)
    response = client.post("/v1/operations/investigations", json=body(use_llm=True))
    assert response.status_code == 503 and "InvestigationPlan" in response.text


def test_demo_counterfactual_changes_diagnosis(repo):
    from ai_agent_template.commerce.operations_demo import demo_reports

    reports = demo_reports()
    for kind in ("supplier", "listing", "ads", "returns", "sales"):
        load(repo, kind, reports[kind])
    query = request(
        sku="SKU-A102",
        asin="DEMO000102",
        comparison_start="2026-08-01",
        comparison_end="2026-08-31",
    )
    analytics = OperationsAnalytics(repo)
    assert analytics.listing(query)["field_differences"]
    assert analytics.ads(query)["anomalies"]
    assert analytics.returns(query)["return_rate"] == 0.08
    # Another seller imports the SAME SKU with corrected source rows, no diagnostic SKU switch.
    for kind in ("supplier", "listing", "ads", "returns", "sales"):
        for row in reports[kind]:
            if kind == "listing":
                row["attributes"]["capacity"] = "30ml"
            if kind == "ads":
                row.update(clicks=100, spend=100, attributed_sales=1000, attributed_orders=50)
            if kind == "returns":
                row["units"] = 20
        load(repo, kind, reports[kind], seller="corrected")
    query = query.model_copy(update={"seller_id": "corrected"})
    assert analytics.listing(query)["field_differences"] == []
    assert analytics.ads(query)["anomalies"] == []
    assert analytics.returns(query)["return_rate"] == 0.02


def test_cross_format_duplicate_and_explicit_row_updates(repo):
    row = ads_row()
    load(repo, "ads", [row])
    from ai_agent_template.commerce.operations_models import AdsPerformance

    csv_row = {key: str(value) for key, value in row.items() if value is not None}
    load(repo, "ads", [csv_row])
    assert len(repo.list(AdsPerformance, "seller", "US")) == 1
    load(repo, "ads", [ads_row(id="source-row-1")])
    load(repo, "ads", [ads_row(id="source-row-1", spend=200)])
    assert len(repo.list(AdsPerformance, "seller", "US")) == 2


def test_actual_csv_xlsx_mapping_and_decimal_normalization(repo):
    import csv
    from io import BytesIO, StringIO

    from openpyxl import Workbook

    from ai_agent_template.commerce.operations_models import AdsPerformance

    row = {key: value for key, value in ads_row().items() if value is not None}
    row["Seller SKU"] = row.pop("sku")
    text = StringIO()
    writer = csv.DictWriter(text, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    import_report(repo, "ads", "seller", "US", text.getvalue().encode(), "actual.csv")
    row["spend"] = "100.00"
    workbook = Workbook()
    workbook.active.append(list(row))
    workbook.active.append(list(row.values()))
    buffer = BytesIO()
    workbook.save(buffer)
    import_report(repo, "ads", "seller", "US", buffer.getvalue(), "actual.xlsx")
    assert len(repo.list(AdsPerformance, "seller", "US")) == 1


def test_bounded_llm_plan_repair(system, monkeypatch):
    from ai_agent_template.agent.decisions import Decision, LLMQueryPolicy
    from ai_agent_template.commerce.operations_models import InvestigationTasks

    client, engine, _ = system
    load(engine.repository, "supplier", [supplier()], seller="demo-store")
    calls = []
    monkeypatch.setattr(engine, "model", lambda: object())

    async def repair(self, schema, *args, **kwargs):
        calls.append(schema)
        if len(calls) == 1:
            return Decision(None, tokens=100, error="invalid_structured_output")
        return Decision(
            InvestigationTasks(
                modules=["listing"],
            ),
            tokens=100,
        )

    monkeypatch.setattr(LLMQueryPolicy, "_call", repair)
    response = client.post("/v1/operations/investigations", json=body(use_llm=True))
    assert response.status_code == 200, response.text
    assert calls == [InvestigationTasks, InvestigationTasks]
    assert response.json()["verification"]["decision_tokens"] == 200


@pytest.mark.asyncio
async def test_semantic_rejection_does_not_publish_causal_claim(monkeypatch):
    from ai_agent_template.agent.decisions import Decision, LLMQueryPolicy
    from ai_agent_template.commerce.analytics import evidence
    from ai_agent_template.commerce.interpretation import Interpretation, interpret
    from ai_agent_template.rag.evidence import ClaimAssessment, ClaimAssessments, GroundedClaim

    ref = evidence(request(), [], "仅有时间共现，因果未确认", {})
    claim = GroundedClaim(text="包装变更确定导致退货上升", evidence_ids=[ref["evidence_id"]])

    async def policy(self, schema, *args, **kwargs):
        if schema is Interpretation:
            return Decision(Interpretation(explanations=[claim]), tokens=100)
        return Decision(
            ClaimAssessments(
                checks=[
                    ClaimAssessment(
                        claim_index=0,
                        claim=claim.text,
                        evidence_ids=claim.evidence_ids,
                        verdict="unsupported",
                        reason="Temporal co-occurrence is not causation",
                    )
                ]
            ),
            tokens=100,
        )

    monkeypatch.setattr(LLMQueryPolicy, "_call", policy)
    result = await interpret(object(), "调查因果", {}, [ref], 4000)
    assert result["findings"] == [] and not result["verification"]["supported"]


def test_history_date_filter_and_no_knowledge(system):
    client, engine, _ = system
    load(engine.repository, "supplier", [supplier()], seller="demo-store")
    response = client.post("/v1/operations/investigations", json=body(modules=["knowledge"]))
    assert response.status_code == 200
    assert response.json()["missing_evidence"] and not response.json()["document_evidence"]


def test_corrected_report_preserves_original_batch_evidence(repo):
    from ai_agent_template.commerce.operations_models import AdsPerformance

    first = load(repo, "ads", [ads_row(id="stable-source-row", spend=100)])
    row_id = repo.list(AdsPerformance, "seller", "US")[0].id
    load(repo, "ads", [ads_row(id="stable-source-row", spend=200)])
    assert repo.source_rows("seller", "US", [row_id])[0]["spend"] == "200"
    assert repo.source_rows("seller", "US", [row_id], [first["id"]])[0]["spend"] == "100"
    assert repo.source_rows("other", "US", [row_id], [first["id"]]) == []


def test_partial_report_still_answers_and_does_not_invent_rates(system):
    client, engine, _ = system
    load(engine.repository, "supplier", [supplier()], seller="demo-store")
    load(
        engine.repository,
        "returns",
        [supplier(attributes=None) | dict(return_date="2026-09-10", units=3, reason="damaged")],
        seller="demo-store",
    )
    response = client.post("/v1/operations/investigations", json=body())
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "3 件退货" in answer and "销售分母缺失" in answer
    assert "无法计算有效退货率" in answer and "0%" not in answer
    assert "没有广告报告" in answer and "尚未检索到" in answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,text",
    [
        ("supported", "报表记录了 3 件退货。"),
        ("unsupported", "包装导致退货。"),
        ("supported", "报表记录了 999 件退货。"),
        ("supported", "退货率低于行业平均水平。"),
    ],
)
async def test_answer_rewrite_requires_semantic_and_numeric_support(monkeypatch, verdict, text):
    from ai_agent_template.agent.decisions import Decision, LLMQueryPolicy
    from ai_agent_template.commerce.analytics import evidence
    from ai_agent_template.commerce.interpretation import Interpretation, interpret
    from ai_agent_template.commerce.operations_models import AnswerParagraph
    from ai_agent_template.rag.evidence import ClaimAssessment, ClaimAssessments, GroundedClaim

    ref = evidence(request(), [], "退货件数", {"returned_units": 3})
    claim = GroundedClaim(text=text, evidence_ids=[ref["evidence_id"]])

    async def policy(self, schema, *args, **kwargs):
        if schema is Interpretation:
            return Decision(Interpretation(answer=[claim]), tokens=100)
        return Decision(
            ClaimAssessments(
                checks=[
                    ClaimAssessment(
                        claim_index=0,
                        claim=text,
                        evidence_ids=claim.evidence_ids,
                        verdict=verdict,
                        reason="test judgement",
                    )
                ]
            ),
            tokens=100,
        )

    monkeypatch.setattr(LLMQueryPolicy, "_call", policy)
    draft = [AnswerParagraph(text="3 件退货", evidence_ids=claim.evidence_ids)]
    result = await interpret(object(), "退货怎么样", {}, [ref], 4000, draft)
    assert bool(result["answer"]) == (
        verdict == "supported" and "999" not in text and "行业" not in text
    )


@pytest.mark.asyncio
async def test_verified_answer_cannot_drop_another_module(monkeypatch):
    from ai_agent_template.agent.decisions import Decision, LLMQueryPolicy
    from ai_agent_template.commerce.analytics import evidence
    from ai_agent_template.commerce.interpretation import Interpretation, interpret
    from ai_agent_template.commerce.operations_models import AnswerParagraph
    from ai_agent_template.rag.evidence import ClaimAssessment, ClaimAssessments, GroundedClaim

    refs = [
        evidence(request(), [], "退货", {"returned_units": 3}),
        evidence(request(), [], "广告", {"spend": 100}),
    ]
    claim = GroundedClaim(text="报表有 3 件退货。", evidence_ids=[refs[0]["evidence_id"]])

    async def policy(self, schema, *args, **kwargs):
        if schema is Interpretation:
            return Decision(Interpretation(answer=[claim]), tokens=100)
        return Decision(
            ClaimAssessments(
                checks=[
                    ClaimAssessment(
                        claim_index=0,
                        claim=claim.text,
                        evidence_ids=claim.evidence_ids,
                        verdict="supported",
                        reason="correct",
                    )
                ]
            ),
            tokens=100,
        )

    monkeypatch.setattr(LLMQueryPolicy, "_call", policy)
    draft = [
        AnswerParagraph(text="3 件退货", evidence_ids=[refs[0]["evidence_id"]]),
        AnswerParagraph(text="广告花费 100", evidence_ids=[refs[1]["evidence_id"]]),
    ]
    result = await interpret(object(), "退货和广告", {}, refs, 4000, draft)
    assert not result["answer"] and not result["verification"]["answer_coverage"]

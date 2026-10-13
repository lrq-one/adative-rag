"""Seller operations contracts. A SKU is never a global product identity."""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Module = Literal["listing", "returns", "ads", "knowledge"]


class OperationsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    seller_id: str
    marketplace_id: str
    sku: str = Field(min_length=1)
    asin: str = ""
    product_id: str = Field(min_length=1)
    product_version: str = "1"
    source_id: str = Field(min_length=1)
    report_period: str = ""
    import_batch_id: str = ""
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_at: date
    expires_at: date | None = None
    version: int = Field(default=1, ge=1)
    synthetic: bool = False

    @model_validator(mode="after")
    def interval(self) -> "OperationsRecord":
        if self.expires_at and self.expires_at <= self.effective_at:
            raise ValueError("expires_at must follow effective_at")
        return self


class ListingSnapshot(OperationsRecord):
    title: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)


class ListingIssue(OperationsRecord):
    field: str
    message: str
    severity: Literal["low", "medium", "high"] = "medium"


class SupplierDocument(OperationsRecord):
    attributes: dict[str, str] = Field(default_factory=dict)
    document_id: str = ""
    document_version: int = 1
    evidence_ids: list[str] = Field(default_factory=list)
    description: str = ""


class ProductRevision(OperationsRecord):
    change_type: str
    description: str
    previous_version: str


class ReturnRecord(OperationsRecord):
    return_date: date
    sold_date: date | None = None
    units: int = Field(ge=0)
    reason: str
    cohort_id: str = ""


class SalesRecord(OperationsRecord):
    sold_date: date
    units: int = Field(ge=0)
    cohort_id: str = ""


class QualityRecord(OperationsRecord):
    inspection_date: date
    batch_id: str
    result: str


class AdsCampaign(OperationsRecord):
    campaign_id: str
    name: str
    ad_type: str = "SP"


class AdsPerformance(OperationsRecord):
    report_date: date
    campaign_id: str
    ad_group_id: str
    search_term: str = ""
    keyword: str = ""
    match_type: str = ""
    ad_type: str = "SP"
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    attribution_window: int = Field(ge=1, le=90)
    impressions: int | None = Field(default=None, ge=0)
    clicks: int | None = Field(default=None, ge=0)
    spend: Decimal | None = Field(default=None, ge=0)
    attributed_sales: Decimal | None = Field(default=None, ge=0)
    attributed_orders: int | None = Field(default=None, ge=0)


class AdsSearchTerm(AdsPerformance):
    search_term: str = Field(min_length=1)


class RuleRecord(OperationsRecord):
    required_fields: list[str]
    authority: Literal["internal", "platform"]
    reference: str = Field(min_length=1)


class BusinessEvidence(BaseModel):
    evidence_id: str
    source_type: str
    seller_id: str
    marketplace: str
    sku: str
    import_batch_id: list[str]
    report_name: list[str]
    row_ids: list[str]
    time_range: dict[str, str]
    metric_definition: str
    source_document_id: str = ""
    source_version: str = ""
    filters: dict[str, Any]
    calculation: dict[str, Any]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seller_id: str
    marketplace_id: str
    sku: str
    asin: str = ""
    product_version: str | None = None
    start: date
    end: date
    comparison_start: date | None = None
    comparison_end: date | None = None
    as_of: date | None = None
    return_basis: Literal["event_period", "sales_cohort"] = "event_period"
    campaign_id: str | None = None
    ad_group_id: str | None = None
    search_term: str | None = None

    @model_validator(mode="after")
    def windows(self) -> "AnalysisRequest":
        if self.end < self.start:
            raise ValueError("Invalid time range")
        if bool(self.comparison_start) != bool(self.comparison_end):
            raise ValueError("Both comparison dates are required")
        if self.comparison_start and self.comparison_end:
            if self.comparison_end < self.comparison_start or self.comparison_end >= self.start:
                raise ValueError("Comparison must be an earlier, nonoverlapping window")
        return self


class InvestigationRequest(AnalysisRequest):
    question: str = Field(min_length=1, max_length=8000)
    # Explicit selection is required for deterministic mode; it is not fake language understanding.
    modules: list[Module] = Field(default_factory=list, max_length=4)
    use_llm: bool = False
    history: list[str] = Field(default_factory=list, max_length=6)


class InvestigationTasks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modules: list[Module] = Field(min_length=1, max_length=4)
    route: Literal["dense", "sparse", "hybrid"] = "hybrid"
    required_facts: list[str] = Field(default_factory=list, max_length=8)
    knowledge_queries: list[str] = Field(default_factory=list, max_length=3)


class InvestigationPlan(InvestigationTasks):
    seller_id: str
    marketplace_id: str
    sku: str
    start: date
    end: date
    asin: str = ""
    product_version: str | None = None


class InvestigationFinding(BaseModel):
    kind: Literal["fact", "correlation", "hypothesis", "suggestion"]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    requires_review: bool = True


class ReturnSummary(BaseModel):
    returned_units: int
    sold_units: int | None
    return_rate: float | None
    reasons: dict[str, int]


class Investigation(BaseModel):
    id: str
    seller_id: str
    request: InvestigationRequest
    plan: InvestigationPlan


class InvestigationReport(BaseModel):
    investigation: Investigation
    results: dict[str, Any]
    findings: list[InvestigationFinding]
    evidence: list[BusinessEvidence]
    document_evidence: list[dict[str, Any]]
    missing_evidence: list[str]
    failed_tools: dict[str, str]
    execution: list[str]
    elapsed_ms: float
    generation_mode: str
    verification: dict[str, Any]
    actions_executed: list[str] = Field(default_factory=list)

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Entity(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    store_id: str = Field(min_length=1, max_length=128)
    updated_at: str = Field(default_factory=now)
    synthetic: bool = False


class Product(Entity):
    brand: str
    category: str
    names: dict[str, str] = Field(min_length=1)


class ProductVariant(Entity):
    product_id: str
    sku: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    attributes: dict[str, str] = Field(default_factory=dict)
    barcode: str | None = None
    price: Decimal | None = Field(default=None, ge=0)
    currency: str = "USD"
    stock: int | None = Field(default=None, ge=0)
    markets: list[str] = Field(default_factory=list)


class ProductDocument(Entity):
    product_id: str
    sku: str
    category: str
    market: str
    language: str
    source: str
    document_id: str
    document_version: int
    evidence_ids: list[str]
    effective_at: str | None = None
    expires_at: str | None = None

    @model_validator(mode="after")
    def dates(self) -> "ProductDocument":
        validate_dates(self.effective_at, self.expires_at)
        return self


class ListingDraft(Entity):
    sku: str
    platform: str
    market: str
    language: str
    title: str
    bullets: list[str]
    description: str
    facts: list[dict[str, Any]]
    issues: list[str]
    review_status: Literal["pending_review", "blocked", "approved"] = "pending_review"
    generation_mode: str


class OrderItem(BaseModel):
    sku: str
    quantity: int = Field(ge=1)


class Order(Entity):
    customer_id: str
    market: str
    payment_status: str
    fulfillment_status: str
    status: str
    items: list[OrderItem]
    placed_at: str
    delivered_at: str | None = None


class Shipment(Entity):
    order_id: str
    tracking_number: str
    carrier: str
    status: str
    events: list[dict[str, str]]


class ReturnCase(Entity):
    order_id: str
    customer_id: str
    reason: str
    status: str = "pending_human_review"
    suggestion: str


class StorePolicy(Entity):
    kind: Literal["shipping", "return", "refund"]
    market: str
    language: str
    text: str
    effective_at: str
    expires_at: str | None = None
    return_days: int = Field(default=30, ge=0, le=365)
    requires_unopened: bool = True
    damaged_exception: bool = True

    @model_validator(mode="after")
    def dates(self) -> "StorePolicy":
        validate_dates(self.effective_at, self.expires_at)
        return self


class Principal(BaseModel):
    store_id: str
    subject: str
    customer_id: str | None = None
    role: Literal["operator", "customer"]
    marketplaces: list[str] = Field(default_factory=lambda: ["US", "DE", "UK"])


class CatalogImport(BaseModel):
    products: list[Product]
    variants: list[ProductVariant]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16000)
    thread_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    skus: list[str] = Field(default_factory=list, max_length=8)
    market: str = "US"
    language: str = "en"
    order_id: str | None = None


class ListingRequest(BaseModel):
    sku: str
    platform: str = "amazon"
    market: str = "US"
    language: str = "en"
    content_types: list[Literal["title", "bullets", "description"]] = Field(
        default=["title", "bullets", "description"], min_length=1, max_length=3
    )


class AfterSalesRequest(BaseModel):
    reason: Literal["damaged", "wrong_item", "missing_item", "changed_mind", "late_delivery"]
    opened: bool | None = None
    received_at: str | None = None


def parsed_date(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Dates require a timezone")
    return result


def validate_dates(start: str | None, end: str | None) -> None:
    if start:
        parsed_date(start)
    if end:
        parsed_date(end)
    if start and end and parsed_date(end) <= parsed_date(start):
        raise ValueError("expires_at must be later than effective_at")


def effective(item: ProductDocument | StorePolicy, at: datetime | None = None) -> bool:
    at = at or datetime.now(timezone.utc)
    return (not item.effective_at or parsed_date(item.effective_at) <= at) and (
        not item.expires_at or parsed_date(item.expires_at) > at
    )

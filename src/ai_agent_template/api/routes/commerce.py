import asyncio
import json
from datetime import datetime, timezone
from functools import lru_cache
from threading import RLock
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ai_agent_template.commerce.orders import after_sales
from ai_agent_template.commerce.schemas import (
    AfterSalesRequest,
    CatalogImport,
    ChatRequest,
    ListingDraft,
    ListingRequest,
    Principal,
    Product,
    ProductDocument,
    ProductVariant,
    StorePolicy,
    effective,
    parsed_date,
)
from ai_agent_template.commerce.service import CommerceService
from ai_agent_template.core.settings import get_settings

router = APIRouter()
_factory_lock = RLock()


def get_commerce_service() -> CommerceService:
    with _factory_lock:
        return _build_commerce_service()


@lru_cache(maxsize=1)
def _build_commerce_service() -> CommerceService:
    return CommerceService(get_settings())


def principal(request: Request) -> Principal:
    settings = get_settings()
    token = request.headers.get("authorization", "").removeprefix("Bearer ")
    trusted = settings.commerce_principals.get(token) if token else None
    if trusted:
        return Principal.model_validate(trusted)
    if not settings.commerce_production:
        demo = {
            "demo-operator": Principal(
                store_id="demo-store", subject="demo-operator", role="operator"
            ),
            "demo-customer": Principal(
                store_id="demo-store",
                subject="demo-customer",
                customer_id="demo-customer",
                role="customer",
            ),
        }
        if token in demo:
            return demo[token]
    raise HTTPException(
        status_code=401, detail="需要有效的店铺身份令牌", headers={"WWW-Authenticate": "Bearer"}
    )


def operator(user: Annotated[Principal, Depends(principal)]) -> Principal:
    if user.role != "operator":
        raise HTTPException(status_code=403, detail="需要运营人员权限")
    return user


User = Annotated[Principal, Depends(principal)]
Operator = Annotated[Principal, Depends(operator)]
Service = Annotated[CommerceService, Depends(get_commerce_service)]


def market_allowed(user: Principal, market: str) -> bool:
    return market in {"all", "GLOBAL"} or market in user.marketplaces


def require_market(user: Principal, market: str) -> None:
    if not market_allowed(user, market):
        raise PermissionError("Marketplace access denied")


@router.get("/status")
def status(user: User, service: Service) -> dict[str, Any]:
    return {
        "store_id": user.store_id,
        "role": user.role,
        "mode": service.settings.commerce_llm_mode,
        "database": service.repository.engine.dialect.name,
        "ready": service.repository.health(),
        "synthetic": not service.settings.commerce_production,
    }


@router.get("/products")
def products(
    user: User,
    service: Service,
    q: str = "",
    sku: str | None = None,
    attribute: str | None = None,
    value: str | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    if market:
        require_market(user, market)
    catalog = {item.id: item for item in service.repository.list(Product, user.store_id)}
    variants = [
        item
        for item in service.repository.list(ProductVariant, user.store_id)
        if (not sku or item.sku == sku)
        and (not market or not item.markets or market in item.markets)
        and (not item.markets or bool(set(item.markets) & set(user.marketplaces)))
        and (not attribute or item.attributes.get(attribute) == value)
        and (
            not q
            or q.casefold()
            in (item.sku + " " + " ".join(catalog[item.product_id].names.values())).casefold()
        )
    ]
    return {
        "products": [
            catalog[item].model_dump()
            for item in sorted({variant.product_id for variant in variants})
        ],
        "variants": [item.model_dump(mode="json") for item in variants],
    }


@router.post("/catalog")
def catalog(request: CatalogImport, user: Operator, service: Service) -> dict[str, int]:
    return service.catalog(user, request)


@router.post("/catalog/import")
async def catalog_file(
    user: Operator, service: Service, file: Annotated[UploadFile, File(...)]
) -> dict[str, Any]:
    return await service.catalog_file(user, await file.read(), file.filename or "catalog.csv")


@router.post("/demo/seed")
async def seed(user: Operator, service: Service) -> dict[str, int]:
    return await service.seed_knowledge(user)


@router.post("/documents")
async def upload(
    user: Operator,
    service: Service,
    file: Annotated[UploadFile, File(...)],
    sku: Annotated[str, Form()],
    market: Annotated[str, Form()] = "US",
    language: Annotated[str, Form()] = "en",
    document_id: Annotated[str | None, Form()] = None,
    effective_at: Annotated[str | None, Form()] = None,
    expires_at: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    require_market(user, market)
    return await service.upload(
        user,
        sku,
        await file.read(),
        file.filename or "upload.txt",
        market,
        language,
        document_id,
        effective_at,
        expires_at,
    )


@router.get("/documents")
def documents(user: User, service: Service, sku: str | None = None) -> list[dict[str, Any]]:
    return [
        {**item.model_dump(), "effective": effective(item)}
        for item in service.repository.list(ProductDocument, user.store_id)
        if not sku or item.sku == sku
        if market_allowed(user, item.market)
    ]


@router.get("/documents/{document_id}/versions")
def versions(document_id: str, user: User, service: Service) -> list[dict[str, Any]]:
    owned = service.repository.get(ProductDocument, user.store_id, document_id)
    if not owned or not market_allowed(user, owned.market):
        raise HTTPException(404, detail="Document not found")
    return [
        item.model_dump() for item in service.rag(user.store_id).repository.versions(document_id)
    ]


@router.get("/documents/{document_id}/{version}/file")
def original(document_id: str, version: int, user: User, service: Service) -> FileResponse:
    owned = service.repository.get(ProductDocument, user.store_id, document_id)
    if not owned or not market_allowed(user, owned.market):
        raise HTTPException(404, detail="Document not found")
    items = service.rag(user.store_id).repository.versions(document_id)
    item = next(
        (
            item
            for item in items
            if item.document_version == version
            and item.index_status in {"active", "historical"}
            and item.original_file
        ),
        None,
    )
    if not item:
        raise HTTPException(404, detail="Original unavailable")
    proof = (
        service.rag(user.store_id).repository.resolve(item.evidence_ids[0])
        if item.evidence_ids
        else None
    )
    if not proof or not market_allowed(user, str(proof[0].metadata.get("market", owned.market))):
        raise HTTPException(404, detail="Original unavailable")
    return FileResponse(str(item.original_file), filename=item.source_filename)


@router.get("/evidence/{evidence_id}")
def evidence(evidence_id: str, user: User, service: Service) -> dict[str, Any]:
    resolved = service.rag(user.store_id).repository.resolve(evidence_id)
    if not resolved or resolved[0].metadata.get("store_id") != user.store_id:
        raise HTTPException(404, detail="Evidence unavailable")
    record, record_status = resolved
    if not market_allowed(user, str(record.metadata.get("market", "all"))):
        raise HTTPException(404, detail="Evidence unavailable")
    doc = service.repository.get(ProductDocument, user.store_id, record.document_id)
    at = datetime.now(timezone.utc)
    expired = bool(
        record.metadata.get("expires_at") and parsed_date(record.metadata["expires_at"]) <= at
    )
    return {
        "evidence": record.model_dump(),
        "status": "expired" if expired or not doc else record_status,
    }


@router.get("/search")
async def search(
    user: User,
    service: Service,
    q: Annotated[str, Query(min_length=1, max_length=2000)],
    sku: str | None = None,
    market: str = "US",
    mode: str = "hybrid",
) -> dict[str, Any]:
    from ai_agent_template.commerce.retrieval import CommerceScope, scope

    if mode not in {"dense", "sparse", "hybrid"}:
        raise HTTPException(400, detail="Invalid route")
    require_market(user, market)
    token = scope.set(CommerceScope(user, market, "en", [sku] if sku else []))
    try:
        from typing import cast

        from ai_agent_template.rag.service import RetrievalMode

        sources = await asyncio.to_thread(
            service.retriever.search, q, mode=cast(RetrievalMode, mode)
        )
        return {
            "sources": [item.model_dump() for item in sources],
            "mode": service.settings.commerce_llm_mode,
        }
    finally:
        scope.reset(token)


@router.post("/listings", response_model=ListingDraft)
async def listing(request: ListingRequest, user: Operator, service: Service) -> ListingDraft:
    require_market(user, request.market)
    return await service.listing(user, request)


@router.get("/listings")
def listings(user: Operator, service: Service) -> list[ListingDraft]:
    return [
        item
        for item in service.repository.list(ListingDraft, user.store_id)
        if market_allowed(user, item.market)
    ]


@router.post("/chat")
async def chat(request: ChatRequest, user: User, service: Service) -> dict[str, Any]:
    require_market(user, request.market)
    return await service.chat(user, request)


@router.post("/chat/stream")
async def stream(request: ChatRequest, user: User, service: Service) -> StreamingResponse:
    require_market(user, request.market)
    # Resolve/validate before returning 200; errors aren't hidden inside a partial SSE stream.
    result = await service.chat(user, request)

    async def events() -> Any:
        yield (
            "event: sources\ndata: "
            + json.dumps({"sources": result.get("sources", [])}, ensure_ascii=False)
            + "\n\n"
        )
        text = result.get("answer") or ""
        for start in range(0, len(text), 256):
            yield (
                "event: delta\ndata: "
                + json.dumps({"text": text[start : start + 256]}, ensure_ascii=False)
                + "\n\n"
            )
        yield "event: done\ndata: " + json.dumps(result, ensure_ascii=False) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/orders/{order_id}")
def order(order_id: str, user: User, service: Service) -> dict[str, Any]:
    return service.orders.query(user, order_id)


@router.post("/orders/{order_id}/after-sales")
def returns(
    order_id: str, request: AfterSalesRequest, user: User, service: Service
) -> dict[str, Any]:
    return after_sales(service.repository, user, order_id, request)


@router.get("/policies")
def policies(user: User, service: Service) -> list[StorePolicy]:
    return [
        item
        for item in service.repository.list(StorePolicy, user.store_id)
        if market_allowed(user, item.market)
    ]


@router.post("/policies")
def policy(request: StorePolicy, user: Operator, service: Service) -> StorePolicy:
    require_market(user, request.market)
    if request.store_id != user.store_id:
        raise HTTPException(403, detail="Store mismatch")
    service.repository.put(request)
    return request

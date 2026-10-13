import json
from collections.abc import AsyncIterator
from datetime import date
from threading import RLock
from typing import Annotated, Any
from weakref import WeakKeyDictionary

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ai_agent_template.api.routes.commerce import Operator, get_commerce_service
from ai_agent_template.commerce.investigation import InvestigationEngine
from ai_agent_template.commerce.operations_demo import seed_operations
from ai_agent_template.commerce.operations_models import (
    AnalysisRequest,
    InvestigationRequest,
    SupplierDocument,
)
from ai_agent_template.commerce.operations_repository import OperationsRepository
from ai_agent_template.commerce.report_importer import import_report
from ai_agent_template.commerce.schemas import Principal
from ai_agent_template.commerce.service import CommerceService
from ai_agent_template.rag.repository import fingerprint

router = APIRouter()
_engines: WeakKeyDictionary[CommerceService, InvestigationEngine] = WeakKeyDictionary()
_lock = RLock()


def get_operations(
    service: Annotated[CommerceService, Depends(get_commerce_service)],
) -> InvestigationEngine:
    with _lock:
        if service not in _engines:
            repository = OperationsRepository(service.repository)
            if service.settings.commerce_seed_demo and not service.settings.commerce_production:
                seed_operations(repository)
            _engines[service] = InvestigationEngine(service, repository)
        return _engines[service]


Engine = Annotated[InvestigationEngine, Depends(get_operations)]


def authorized_market(user: Principal, seller: str, market: str) -> None:
    if seller != user.store_id or market not in user.marketplaces:
        raise PermissionError("Seller/marketplace access denied")


@router.get("/scope")
def scope(user: Operator) -> dict[str, Any]:
    return dict(sellers=[user.store_id], marketplaces=user.marketplaces)


@router.get("/rows")
def evidence_rows(
    user: Operator, engine: Engine, marketplace_id: str, row_ids: str, batch_ids: str | None = None
) -> list[dict[str, Any]]:
    authorized_market(user, user.store_id, marketplace_id)
    return engine.repository.source_rows(
        user.store_id,
        marketplace_id,
        row_ids.split(","),
        batch_ids.split(",") if batch_ids else None,
    )


@router.get("/products")
def products(user: Operator, engine: Engine, marketplace_id: str = "US") -> list[dict[str, Any]]:
    authorized_market(user, user.store_id, marketplace_id)
    values = engine.repository.list(SupplierDocument, user.store_id, marketplace_id)
    unique = {(item.sku, item.asin, item.product_id): item for item in values}
    return [
        dict(
            sku=item.sku,
            asin=item.asin,
            product_id=item.product_id,
            product_version=item.product_version,
            synthetic=item.synthetic,
        )
        for item in unique.values()
    ]


@router.post("/imports/{kind}")
async def reports(
    kind: str,
    user: Operator,
    engine: Engine,
    file: Annotated[UploadFile, File()],
    seller_id: Annotated[str, Form()],
    marketplace_id: Annotated[str, Form()],
    field_mapping: Annotated[str, Form()] = "{}",
) -> dict[str, Any]:
    authorized_market(user, seller_id, marketplace_id)
    mapping = json.loads(field_mapping)
    if not isinstance(mapping, dict) or any(
        not isinstance(value, str) for value in mapping.values()
    ):
        raise ValueError("field_mapping must map original header to canonical field")
    return import_report(
        engine.repository,
        kind,
        seller_id,
        marketplace_id,
        await file.read(),
        file.filename or "report.csv",
        mapping,
    )


@router.post("/analysis/{module}")
async def analysis(
    module: str, request: AnalysisRequest, user: Operator, engine: Engine
) -> dict[str, Any]:
    authorized_market(user, request.seller_id, request.marketplace_id)
    if module not in {"listing", "returns", "ads"}:
        raise HTTPException(404, "Unknown module")
    import asyncio

    return await asyncio.to_thread(getattr(engine.analytics, module), request)


@router.post("/investigations")
async def investigate(
    request: InvestigationRequest, user: Operator, engine: Engine
) -> dict[str, Any]:
    return (await engine.run(user, request)).model_dump(mode="json")


@router.post("/investigations/stream")
async def investigation_stream(
    request: InvestigationRequest, user: Operator, engine: Engine
) -> StreamingResponse:
    engine.authorize(user, request)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in engine.stream(user, request):
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:
            # No exception details containing request headers or provider credentials.
            yield (
                "event: error\ndata: "
                + json.dumps(
                    dict(
                        type="error",
                        message="诊断失败，请检查配置或数据并重试",
                        error_type=type(error).__name__,
                    ),
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/investigations/{report_id}")
def investigation_report(report_id: str, user: Operator, engine: Engine) -> dict[str, Any]:
    result = engine.repository.report(user.store_id, report_id)
    if result is None:
        raise LookupError("Investigation not found")
    authorized_market(user, user.store_id, result["investigation"]["request"]["marketplace_id"])
    return result


@router.post("/documents")
async def upload(
    user: Operator,
    engine: Engine,
    file: Annotated[UploadFile, File()],
    seller_id: Annotated[str, Form()],
    marketplace_id: Annotated[str, Form()],
    sku: Annotated[str, Form()],
    asin: Annotated[str, Form()],
    product_id: Annotated[str, Form()],
    effective_at: Annotated[date, Form()],
    expires_at: Annotated[date | None, Form()] = None,
    product_version: Annotated[str, Form()] = "1",
    attributes: Annotated[str, Form()] = "{}",
    source_key: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    authorized_market(user, seller_id, marketplace_id)
    request = AnalysisRequest(
        seller_id=seller_id,
        marketplace_id=marketplace_id,
        sku=sku,
        asin=asin,
        start=effective_at,
        end=effective_at,
    )
    filename = file.filename or "supplier.md"
    validated_attributes = json.loads(attributes)
    # Validate before indexing; malformed SQL metadata must not leave a published upload.
    SupplierDocument(
        seller_id=seller_id,
        marketplace_id=marketplace_id,
        sku=sku,
        asin=asin,
        product_id=product_id,
        product_version=product_version,
        source_id=filename,
        effective_at=effective_at,
        attributes=validated_attributes,
        expires_at=expires_at,
    )
    content = await file.read()
    result = await engine.knowledge.upload(request, content, filename, product_version, source_key)
    batch = dict(
        id=fingerprint(
            [
                seller_id,
                marketplace_id,
                sku,
                asin,
                product_id,
                product_version,
                validated_attributes,
                content.hex(),
                effective_at.isoformat(),
                str(expires_at),
                result["document_version"],
            ]
        ),
        seller_id=seller_id,
        filename=filename,
    )
    item = SupplierDocument(
        seller_id=seller_id,
        marketplace_id=marketplace_id,
        sku=sku,
        asin=asin,
        product_id=product_id,
        product_version=product_version,
        source_id=filename,
        effective_at=effective_at,
        id=fingerprint([sku, asin, product_id, result["document_id"], effective_at.isoformat()]),
        expires_at=expires_at,
        import_batch_id=batch["id"],
        attributes=validated_attributes,
        document_id=result["document_id"],
        document_version=result["document_version"],
    )
    manifest = engine.knowledge.rag(request).repository.active(item.document_id)
    item.evidence_ids = manifest.evidence_ids if manifest else []
    engine.repository.import_rows("supplier", [item], batch)
    return dict(document=item.model_dump(mode="json"), index=result)


@router.post("/knowledge/search")
async def search(request: InvestigationRequest, user: Operator, engine: Engine) -> dict[str, Any]:
    engine.authorize(user, request)
    return await engine.knowledge.retrieve(
        request, [request.question], "hybrid", engine.model() if request.use_llm else None
    )


@router.get("/documents/{document_id}/{version}/file")
def original(
    document_id: str,
    version: int,
    user: Operator,
    engine: Engine,
    marketplace_id: str,
    sku: str,
    asin: str = "",
) -> FileResponse:
    authorized_market(user, user.store_id, marketplace_id)
    request = AnalysisRequest(
        seller_id=user.store_id,
        marketplace_id=marketplace_id,
        sku=sku,
        asin=asin,
        start=date.today(),
        end=date.today(),
    )
    rag = engine.knowledge.rag(request)
    manifest = next(
        (
            item
            for item in rag.repository.versions(document_id)
            if item.document_version == version and item.namespace == rag._settings.rag_namespace
        ),
        None,
    )
    if not manifest or not manifest.original_file:
        raise LookupError("Original not found")
    return FileResponse(manifest.original_file, filename=manifest.source_filename)

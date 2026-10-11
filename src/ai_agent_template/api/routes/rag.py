import asyncio
import time
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from ai_agent_template.api.dependencies import get_rag_service
from ai_agent_template.core.settings import get_settings
from ai_agent_template.rag.indexing import logical_id
from ai_agent_template.rag.loaders import load_upload_documents, parser_identity, save_original
from ai_agent_template.rag.repository import digest
from ai_agent_template.rag.schemas import (
    IndexStats,
    RagIngestRequest,
    RagIngestResponse,
    RagSearchResponse,
)
from ai_agent_template.rag.service import RagService

router = APIRouter()
RagServiceDependency = Annotated[RagService, Depends(get_rag_service)]


@router.post("/documents", response_model=RagIngestResponse)
async def ingest_document(
    request: RagIngestRequest,
    rag_service: RagServiceDependency,
) -> RagIngestResponse:
    return await _ingest_text(rag_service, request, request.document_type)


async def _ingest_text(
    rag_service: RagService, request: RagIngestRequest, kind: Any
) -> RagIngestResponse:
    if request.namespace != get_settings().rag_namespace:
        raise HTTPException(
            status_code=400, detail="Namespace differs from this application's configured namespace"
        )
    try:
        return await asyncio.to_thread(
            rag_service.ingest_text,
            text=request.text,
            source=request.source,
            document_type=kind,
            metadata=request.metadata,
            document_id=request.document_id,
            namespace=request.namespace,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/files", response_model=RagIngestResponse)
async def ingest_file(
    rag_service: RagServiceDependency,
    file: Annotated[UploadFile, File(...)],
    document_id: Annotated[str | None, Form(max_length=128)] = None,
    namespace: Annotated[str, Form(max_length=128)] = "default",
) -> RagIngestResponse:
    return await _ingest_file(rag_service, file, "knowledge", document_id, namespace)


async def _ingest_file(
    rag_service: Any, file: Any, kind: Any, document_id: Any, namespace: Any
) -> RagIngestResponse:
    settings = get_settings()
    if namespace != settings.rag_namespace:
        raise HTTPException(
            status_code=400, detail="Namespace differs from this application's configured namespace"
        )
    stats = IndexStats()
    started = time.perf_counter()
    content = await file.read()
    owner = (
        document_id
        if document_id and rag_service.repository.versions(document_id)
        else logical_id(namespace, file.filename or "upload", kind, document_id)
    )
    with rag_service.indexer.cache.bind(owner):
        try:
            sections = await load_upload_documents(
                UploadFile(file=BytesIO(content), filename=file.filename),
                settings,
                rag_service.indexer.cache,
                stats,
            )
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=400, detail="File must be UTF-8 text or PDF."
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if not sections:
            raise HTTPException(status_code=400, detail="File did not contain extractable content.")

        original_file = save_original(
            content, Path(file.filename or "upload").suffix.lower(), settings.rag_data_path
        )
        stats.parsing_time_ms = (time.perf_counter() - started) * 1000
        return await _index_sections(
            rag_service,
            sections=sections,
            source=file.filename or "upload",
            metadata={"content_type": file.content_type},
            document_type=kind,
            document_id=document_id,
            namespace=namespace,
            file_hash=digest(content),
            original_file=original_file,
            preprocessing_stats=stats,
            parser_version=parser_identity(settings),
        )


async def _index_sections(rag_service: RagService, **kwargs: Any) -> RagIngestResponse:
    try:
        return await asyncio.to_thread(rag_service.ingest_sections, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/voice-samples", response_model=RagIngestResponse)
async def ingest_voice_sample(
    request: RagIngestRequest,
    rag_service: RagServiceDependency,
) -> RagIngestResponse:
    return await _ingest_text(rag_service, request, "voice_sample")


@router.post("/voice-files", response_model=RagIngestResponse)
async def ingest_voice_file(
    rag_service: RagServiceDependency,
    file: Annotated[UploadFile, File(...)],
    document_id: Annotated[str | None, Form(max_length=128)] = None,
    namespace: Annotated[str, Form(max_length=128)] = "default",
) -> RagIngestResponse:
    return await _ingest_file(rag_service, file, "voice_sample", document_id, namespace)


@router.get("/search", response_model=RagSearchResponse)
async def search(
    rag_service: RagServiceDependency,
    q: Annotated[str, Query(min_length=1, max_length=2_000)],
) -> RagSearchResponse:
    return RagSearchResponse(query=q, sources=rag_service.search(q))


@router.get("/voice/search", response_model=RagSearchResponse)
async def search_voice(
    rag_service: RagServiceDependency,
    q: Annotated[str, Query(min_length=1, max_length=2_000)],
) -> RagSearchResponse:
    return RagSearchResponse(query=q, sources=rag_service.search_voice(q))


@router.delete("/documents/{source}")
async def delete_document(source: str, rag_service: RagServiceDependency) -> dict[str, int | str]:
    """Remove all indexed chunks for a source from Qdrant."""
    return {"source": source, "deleted_chunks": rag_service.delete_source(source)}


@router.get("/evidence/{evidence_id}")
async def get_evidence(evidence_id: str, rag_service: RagServiceDependency) -> dict[str, Any]:
    resolved = rag_service.repository.resolve(evidence_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Evidence version is unavailable or deleted")
    record, status = resolved
    if record.metadata.get("namespace", "default") != get_settings().rag_namespace:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return {"evidence": record.model_dump(), "status": status}


@router.get("/document-versions/{document_id}")
async def get_versions(document_id: str, rag_service: RagServiceDependency) -> list[dict[str, Any]]:
    versions = rag_service.repository.versions(document_id)
    if not versions or versions[0].namespace != get_settings().rag_namespace:
        raise HTTPException(status_code=404, detail="Document not found")
    return [version.model_dump() for version in versions]


@router.get("/index-jobs")
async def get_index_jobs(
    rag_service: RagServiceDependency, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[dict[str, Any]]:
    return rag_service.repository.jobs(limit, get_settings().rag_namespace)


@router.post("/document-versions/{document_id}/{version}/activate")
async def activate_version(
    document_id: str, version: int, rag_service: RagServiceDependency
) -> dict[str, Any]:
    try:
        manifest = await asyncio.to_thread(
            rag_service.indexer.activate, document_id, version, get_settings().rag_namespace
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return manifest.model_dump()


@router.get("/document-versions/{document_id}/{version}/file")
async def get_original(document_id: str, version: int, rag_service: RagServiceDependency) -> Any:
    manifests = rag_service.repository.versions(document_id)
    item = next(
        (
            item
            for item in manifests
            if item.document_version == version
            and item.namespace == get_settings().rag_namespace
            and item.index_status in {"active", "historical"}
        ),
        None,
    )
    if not item or not item.original_file or not Path(item.original_file).is_file():
        raise HTTPException(status_code=404, detail="Original file unavailable")
    return FileResponse(item.original_file, filename=item.source_filename)

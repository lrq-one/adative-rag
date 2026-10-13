import asyncio

from fastapi import APIRouter, HTTPException

from ai_agent_template.api.schemas import HealthResponse
from ai_agent_template.core.settings import get_settings

router = APIRouter()


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    if get_settings().commerce_production:
        from ai_agent_template.api.routes.commerce import get_commerce_service

        try:
            service = await asyncio.to_thread(get_commerce_service)
            await asyncio.to_thread(service.repository.health)
        except Exception as error:
            raise HTTPException(
                status_code=503, detail="Commerce database is unavailable or misconfigured"
            ) from error
    return HealthResponse(status="ok")

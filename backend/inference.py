from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

try:
    from .db import get_db
    from .models import APIKey, Deployment
except ImportError:
    from db import get_db
    from models import APIKey, Deployment

router = APIRouter(tags=["inference"])

_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=5.0)


def _resolve_api_key(request: Request, db: Session) -> APIKey:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        key_value = auth[7:].strip()
    else:
        key_value = auth.strip()

    if not key_value.startswith("ax-"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key format")

    row = db.query(APIKey).filter(APIKey.key == key_value).first()
    if row is None or row.status not in ("trial", "active"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    if row.expires_at is not None and row.expires_at <= datetime.now(timezone.utc).replace(tzinfo=None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")
    return row


def _get_inference_url(api_key: APIKey, db: Session) -> str:
    deployment = (
        db.query(Deployment)
        .filter(
            Deployment.api_key_id == api_key.id,
            Deployment.status == "active",
            Deployment.inference_url.isnot(None),
        )
        .order_by(Deployment.id.desc())
        .first()
    )
    if deployment is None or not deployment.inference_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active deployment found. Deploy a node via the dashboard first.",
        )
    return str(deployment.inference_url).rstrip("/")


async def _proxy(request: Request, upstream_url: str) -> StreamingResponse:
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "content-length")
    }

    async def stream_response():
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream(
                request.method,
                upstream_url,
                content=body,
                headers=headers,
            ) as resp:
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    yield chunk

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.request(
            request.method,
            upstream_url,
            content=body,
            headers=headers,
        )

    content_type = resp.headers.get("content-type", "application/json")
    if "text/event-stream" in content_type:
        return StreamingResponse(stream_response(), media_type=content_type, status_code=resp.status_code)
    return StreamingResponse(
        iter([resp.content]),
        media_type=content_type,
        status_code=resp.status_code,
    )


@router.post("/v1/completions")
@router.post("/v1/chat/completions")
async def proxy_completions(
    request: Request,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    api_key = _resolve_api_key(request, db)
    base_url = _get_inference_url(api_key, db)
    path = request.url.path  # e.g. /v1/completions
    upstream = f"{base_url}{path}"
    return await _proxy(request, upstream)


@router.get("/v1/models")
async def proxy_models(
    request: Request,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    api_key = _resolve_api_key(request, db)
    base_url = _get_inference_url(api_key, db)
    upstream = f"{base_url}/v1/models"
    return await _proxy(request, upstream)

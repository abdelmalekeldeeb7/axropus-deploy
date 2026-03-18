from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from .auth import router as auth_router
from .benchmarks import router as benchmarks_router
from .config import get_settings
from .db import SessionLocal, init_db
from .billing import router as billing_router
from .dashboard import router as dashboard_router
from .deploy import router as deploy_router
from .inference import router as inference_router
from .keys import router as keys_router
from .metrics import router as metrics_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Axropus Cloud Backend", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/health/live")
    def health_live() -> dict:
        return {"status": "live"}

    @app.get("/health/ready")
    def health_ready():
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return {"status": "ready"}
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        finally:
            db.close()

    app.include_router(auth_router)
    app.include_router(benchmarks_router)
    app.include_router(keys_router)
    app.include_router(deploy_router)
    app.include_router(inference_router)
    app.include_router(metrics_router)
    app.include_router(dashboard_router)
    app.include_router(billing_router)
    return app


app = create_app()

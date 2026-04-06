from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

try:
    from .auth import router as auth_router
    from .benchmarks import router as benchmarks_router
    from .claws import router as claws_router
    from .config import get_settings
    from .db import SessionLocal, init_db
    from .billing import router as billing_router
    from .dashboard import router as dashboard_router
    from .deploy import router as deploy_router
    from .economics import router as economics_router
    from .inference import router as inference_router
except ImportError:
    from auth import router as auth_router
    from benchmarks import router as benchmarks_router
    from claws import router as claws_router
    from config import get_settings
    from db import SessionLocal, init_db
    from billing import router as billing_router
    from dashboard import router as dashboard_router
    from deploy import router as deploy_router
    from economics import router as economics_router
    from inference import router as inference_router
try:
    from .keys import router as keys_router
    from .metrics import router as metrics_router
    from .model_manager import model_manager
    from .model_registry import get_model, list_models, list_families
except ImportError:
    from keys import router as keys_router
    from metrics import router as metrics_router
    from model_manager import model_manager
    from model_registry import get_model, list_models, list_families


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield
    await model_manager.shutdown()


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

    # ── Model registry endpoints ────────────────────────────────────────
    @app.get("/v1/models/registry", tags=["models"])
    def list_model_registry(
        family: str | None = None,
        tag: str | None = None,
        supports_openclaw: bool | None = None,
        supports_nemoclaw: bool | None = None,
    ) -> list[dict]:
        """List all models in the Axropus model registry, with optional filters."""
        specs = list_models(
            family=family,
            tag=tag,
            supports_openclaw=supports_openclaw,
            supports_nemoclaw=supports_nemoclaw,
        )
        return [s.to_dict() for s in specs]

    @app.get("/v1/models/registry/{model_id}", tags=["models"])
    def get_model_detail(model_id: str) -> dict:
        """Get detailed metadata for a specific model."""
        spec = get_model(model_id)
        if spec is None:
            return JSONResponse({"detail": f"Model {model_id!r} not found"}, status_code=404)
        return spec.to_dict()

    @app.get("/v1/models/families", tags=["models"])
    def get_model_families() -> list[str]:
        """List all available model families."""
        return list_families()

    # ── Model management endpoints ────────────────────────────────────────
    @app.get("/v1/models/deployments", tags=["models"])
    def list_model_deployments() -> list[dict]:
        """List all managed model deployments and their status."""
        return model_manager.list_deployed_models()

    @app.get("/v1/models/deployments/{deployment_id}", tags=["models"])
    def get_model_deployment_status(deployment_id: int) -> dict:
        """Get status of a specific model deployment."""
        return model_manager.get_model_status(deployment_id)

    # ── Existing routers ──────────────────────────────────────────────────
    app.include_router(auth_router)
    app.include_router(benchmarks_router)
    app.include_router(keys_router)
    app.include_router(deploy_router)
    app.include_router(inference_router)
    app.include_router(metrics_router)
    app.include_router(dashboard_router)
    app.include_router(billing_router)
    app.include_router(claws_router)
    app.include_router(economics_router)
    return app


app = create_app()

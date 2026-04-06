"""OpenClaw agent CRUD and execution — create, configure, run, and monitor AI agents.

Provides a full REST API for managing "Claws" — autonomous AI agents backed by
models from the Axropus model registry. Each Claw binds a model deployment to
a system prompt, tool set, communication channels, and AMF configuration.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:
    from .auth import get_current_customer
    from .db import get_db
    from .models import Claw, ClawTask, ClawStep, Customer
except ImportError:
    from auth import get_current_customer
    from db import get_db
    from models import Claw, ClawTask, ClawStep, Customer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["claws"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ClawCreate(BaseModel):
    """Request body for creating a new Claw agent."""

    name: str = Field(..., min_length=1, max_length=128)
    model_id: str = Field(..., min_length=1, max_length=64)
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    tools: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    amf_config: dict[str, Any] = Field(default_factory=dict)


class ClawUpdate(BaseModel):
    """Request body for updating an existing Claw agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=128)
    model_id: Optional[str] = Field(None, min_length=1, max_length=64)
    system_prompt: Optional[str] = None
    tools: Optional[list[str]] = None
    channels: Optional[list[str]] = None
    openclaw_config: Optional[dict[str, Any]] = None
    amf_config: Optional[dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(active|paused|disabled)$")


class TaskRunRequest(BaseModel):
    """Request body for executing a task on a Claw."""

    prompt: str = Field(..., min_length=1, max_length=32_000)
    context: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=10, le=3600)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _claw_to_dict(claw: Claw) -> dict[str, Any]:
    """Serialize a Claw ORM object to a JSON-safe dict."""
    import json as _json

    def _parse_json_field(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return _json.loads(value)
        except (TypeError, ValueError):
            return value

    return {
        "id": claw.id,
        "customer_id": claw.customer_id,
        "name": claw.name,
        "model_id": claw.model_id,
        "system_prompt": claw.system_prompt,
        "tools": _parse_json_field(claw.tools),
        "channels": _parse_json_field(claw.channels),
        "openclaw_config": _parse_json_field(claw.openclaw_config),
        "amf_config": _parse_json_field(claw.amf_config),
        "status": claw.status,
        "created_at": str(claw.created_at) if claw.created_at else None,
        "updated_at": str(claw.updated_at) if claw.updated_at else None,
    }


def _task_to_dict(task: ClawTask) -> dict[str, Any]:
    """Serialize a ClawTask ORM object to a JSON-safe dict."""
    return {
        "id": task.id,
        "claw_id": task.claw_id,
        "task_uid": task.task_uid,
        "prompt": task.prompt,
        "status": task.status,
        "result": task.result,
        "total_steps": task.total_steps,
        "tokens_used": task.tokens_used,
        "tokens_saved": task.tokens_saved,
        "duration_ms": task.duration_ms,
        "created_at": str(task.created_at) if task.created_at else None,
        "completed_at": str(task.completed_at) if task.completed_at else None,
    }


def _get_claw_or_404(
    claw_id: int,
    customer: Customer,
    db: Session,
) -> Claw:
    """Fetch a Claw belonging to the authenticated customer, or raise 404."""
    row = (
        db.query(Claw)
        .filter(Claw.id == claw_id, Claw.customer_id == customer.id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claw not found")
    return row


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/v1/claws", status_code=status.HTTP_201_CREATED)
def create_claw(
    payload: ClawCreate,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Create a new OpenClaw agent."""
    import json as _json

    claw = Claw(
        customer_id=customer.id,
        name=payload.name.strip(),
        model_id=payload.model_id.strip(),
        system_prompt=payload.system_prompt,
        tools=_json.dumps(payload.tools),
        channels=_json.dumps(payload.channels),
        openclaw_config=_json.dumps(payload.openclaw_config),
        amf_config=_json.dumps(payload.amf_config),
        status="active",
    )
    db.add(claw)
    db.commit()
    db.refresh(claw)
    logger.info("Created claw %d (%s) for customer %d", claw.id, claw.name, customer.id)
    return _claw_to_dict(claw)


@router.get("/v1/claws")
def list_claws(
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List all Claw agents belonging to the authenticated customer."""
    rows = (
        db.query(Claw)
        .filter(Claw.customer_id == customer.id)
        .order_by(Claw.created_at.desc(), Claw.id.desc())
        .all()
    )
    return [_claw_to_dict(c) for c in rows]


@router.get("/v1/claws/{claw_id}")
def get_claw(
    claw_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Get details of a specific Claw agent."""
    claw = _get_claw_or_404(claw_id, customer, db)
    return _claw_to_dict(claw)


@router.put("/v1/claws/{claw_id}")
def update_claw(
    claw_id: int,
    payload: ClawUpdate,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Update a Claw agent's configuration."""
    import json as _json

    claw = _get_claw_or_404(claw_id, customer, db)

    if payload.name is not None:
        claw.name = payload.name.strip()
    if payload.model_id is not None:
        claw.model_id = payload.model_id.strip()
    if payload.system_prompt is not None:
        claw.system_prompt = payload.system_prompt
    if payload.tools is not None:
        claw.tools = _json.dumps(payload.tools)
    if payload.channels is not None:
        claw.channels = _json.dumps(payload.channels)
    if payload.openclaw_config is not None:
        claw.openclaw_config = _json.dumps(payload.openclaw_config)
    if payload.amf_config is not None:
        claw.amf_config = _json.dumps(payload.amf_config)
    if payload.status is not None:
        claw.status = payload.status
    claw.updated_at = _utcnow_naive()

    db.commit()
    db.refresh(claw)
    logger.info("Updated claw %d (%s)", claw.id, claw.name)
    return _claw_to_dict(claw)


@router.delete("/v1/claws/{claw_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_claw(
    claw_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> None:
    """Delete a Claw agent and all associated tasks."""
    claw = _get_claw_or_404(claw_id, customer, db)

    # Delete associated steps and tasks first
    task_ids = [t.id for t in db.query(ClawTask).filter(ClawTask.claw_id == claw.id).all()]
    if task_ids:
        db.query(ClawStep).filter(ClawStep.task_id.in_(task_ids)).delete(synchronize_session=False)
    db.query(ClawTask).filter(ClawTask.claw_id == claw.id).delete(synchronize_session=False)
    db.delete(claw)
    db.commit()
    logger.info("Deleted claw %d", claw_id)


@router.post("/v1/claws/{claw_id}/run")
def run_claw_task(
    claw_id: int,
    payload: TaskRunRequest,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Execute a task using the specified Claw agent.

    Creates a task record, then executes the agent loop (up to *max_steps*).
    In production, this would dispatch to an async worker; here we create the
    task entry and return immediately with a ``pending`` status for the caller
    to poll via the tasks endpoint.
    """
    claw = _get_claw_or_404(claw_id, customer, db)

    if claw.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Claw is {claw.status}; only active claws can run tasks",
        )

    task_uid = str(uuid.uuid4())
    task = ClawTask(
        claw_id=claw.id,
        task_uid=task_uid,
        prompt=payload.prompt,
        status="pending",
        total_steps=0,
        tokens_used=0,
        tokens_saved=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    logger.info("Created task %s for claw %d", task_uid, claw.id)

    # In a production system, this would enqueue an async job.
    # For now, mark as running and return — the task executor will pick it up.
    task.status = "running"
    db.commit()

    return {
        "task_id": task.id,
        "task_uid": task_uid,
        "claw_id": claw.id,
        "status": task.status,
        "message": "Task queued for execution",
    }


@router.get("/v1/claws/{claw_id}/tasks")
def list_claw_tasks(
    claw_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List tasks for a specific Claw agent with pagination."""
    claw = _get_claw_or_404(claw_id, customer, db)

    total = db.query(ClawTask).filter(ClawTask.claw_id == claw.id).count()
    rows = (
        db.query(ClawTask)
        .filter(ClawTask.claw_id == claw.id)
        .order_by(ClawTask.created_at.desc(), ClawTask.id.desc())
        .offset(max(0, offset))
        .limit(min(max(1, limit), 200))
        .all()
    )
    return {
        "claw_id": claw.id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "tasks": [_task_to_dict(t) for t in rows],
    }


@router.get("/v1/claws/{claw_id}/metrics")
def get_claw_metrics(
    claw_id: int,
    customer: Customer = Depends(get_current_customer),
    db: Session = Depends(get_db),
) -> dict:
    """Get aggregated performance metrics for a Claw agent.

    Returns token usage, savings from AMF, task success rates,
    and average execution times.
    """
    claw = _get_claw_or_404(claw_id, customer, db)

    tasks = db.query(ClawTask).filter(ClawTask.claw_id == claw.id).all()

    total_tasks = len(tasks)
    completed = sum(1 for t in tasks if t.status == "completed")
    failed = sum(1 for t in tasks if t.status == "failed")
    running = sum(1 for t in tasks if t.status == "running")
    total_tokens = sum(int(t.tokens_used or 0) for t in tasks)
    total_saved = sum(int(t.tokens_saved or 0) for t in tasks)
    total_steps = sum(int(t.total_steps or 0) for t in tasks)
    durations = [int(t.duration_ms or 0) for t in tasks if t.duration_ms and t.duration_ms > 0]
    avg_duration_ms = sum(durations) / len(durations) if durations else 0.0

    success_rate = (completed / total_tasks * 100.0) if total_tasks > 0 else 0.0
    savings_pct = (total_saved / total_tokens * 100.0) if total_tokens > 0 else 0.0

    return {
        "claw_id": claw.id,
        "claw_name": claw.name,
        "total_tasks": total_tasks,
        "completed": completed,
        "failed": failed,
        "running": running,
        "success_rate_pct": round(success_rate, 2),
        "total_tokens_used": total_tokens,
        "total_tokens_saved": total_saved,
        "amf_savings_pct": round(savings_pct, 2),
        "total_steps": total_steps,
        "avg_duration_ms": round(avg_duration_ms, 1),
    }

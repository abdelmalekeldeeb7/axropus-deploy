from __future__ import annotations

from sqlalchemy import Date, DateTime, ForeignKey, Integer, REAL, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from .db import Base
except ImportError:
    from db import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    company_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    api_keys = relationship("APIKey", back_populates="customer", cascade="all,delete-orphan")
    deployments = relationship("Deployment", back_populates="customer", cascade="all,delete-orphan")
    invoices = relationship("Invoice", back_populates="customer", cascade="all,delete-orphan")
    model_deployments = relationship("ModelDeployment", back_populates="customer", cascade="all,delete-orphan")
    claws = relationship("Claw", back_populates="customer", cascade="all,delete-orphan")
    economics_daily = relationship("EconomicsDaily", back_populates="customer", cascade="all,delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="trial", nullable=False)
    tier: Mapped[str] = mapped_column(String(16), default="trial", nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    expires_at: Mapped[str | None] = mapped_column(DateTime(timezone=False), nullable=True)

    customer = relationship("Customer", back_populates="api_keys")
    metrics = relationship("Metric", back_populates="api_key", cascade="all,delete-orphan")
    deployments = relationship("Deployment", back_populates="api_key_ref")


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True, index=True)
    runtime: Mapped[str] = mapped_column(String(32), nullable=False)
    model_family: Mapped[str] = mapped_column(String(64), nullable=False)
    model_size: Mapped[str] = mapped_column(String(64), nullable=False)
    draft_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    inference_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    deployed_at: Mapped[str | None] = mapped_column(DateTime(timezone=False), nullable=True)

    customer = relationship("Customer", back_populates="deployments")
    api_key_ref = relationship("APIKey", back_populates="deployments")


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    api_key_id: Mapped[int] = mapped_column(ForeignKey("api_keys.id"), nullable=False, index=True)
    timestamp: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    tokens_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    interval_seconds: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    prefix_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    decode_accelerated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    amf_hit_rate: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    spec_acceptance_rate: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    effective_tps: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    baseline_tps: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    compute_saved_pct: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    model_family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_size_bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adapter_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sdk_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    license_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    api_key = relationship("APIKey", back_populates="metrics")


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    period_start: Mapped[str | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[str | None] = mapped_column(Date, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    amount_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="invoices")


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quant_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tensor_parallel: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    gpu_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    deployed_at: Mapped[str | None] = mapped_column(DateTime(timezone=False), nullable=True)
    stopped_at: Mapped[str | None] = mapped_column(DateTime(timezone=False), nullable=True)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="model_deployments")


class Claw(Base):
    __tablename__ = "claws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="You are a helpful AI assistant.", nullable=False)
    tools: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    channels: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    openclaw_config: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    amf_config: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="claws")
    tasks = relationship("ClawTask", back_populates="claw", cascade="all,delete-orphan")


class ClawTask(Base):
    __tablename__ = "claw_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    claw_id: Mapped[int] = mapped_column(ForeignKey("claws.id"), nullable=False, index=True)
    task_uid: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_steps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    completed_at: Mapped[str | None] = mapped_column(DateTime(timezone=False), nullable=True)

    claw = relationship("Claw", back_populates="tasks")
    steps = relationship("ClawStep", back_populates="task", cascade="all,delete-orphan")


class ClawStep(Base):
    __tablename__ = "claw_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("claw_tasks.id"), nullable=False, index=True)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    amf_hit: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    task = relationship("ClawTask", back_populates="steps")


class EconomicsDaily(Base):
    __tablename__ = "economics_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False, index=True)
    date: Mapped[str] = mapped_column(Text, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    axropus_cost_usd: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    openai_equivalent_usd: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    anthropic_equivalent_usd: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    together_equivalent_usd: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    amf_hit_rate: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    prefix_reuse_rate: Mapped[float] = mapped_column(REAL, default=0.0, nullable=False)
    total_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )

    customer = relationship("Customer", back_populates="economics_daily")

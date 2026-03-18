from __future__ import annotations

from sqlalchemy import Date, DateTime, ForeignKey, Integer, REAL, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


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

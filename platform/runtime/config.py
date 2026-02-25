from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ..artifacts.adapter import LocalArtifactStoreAdapter
from ..artifacts.minio_store import MinioArtifactStore
from ..artifacts.s3_store import S3ArtifactStore
from ..ledger.postgres_store import PostgresLedgerStore
from ..ledger.store import SQLiteLedgerStore
from ..queue.inproc_queue import InProcQueue
from ..queue.sqlite_queue import SQLiteQueue
from ..queue.redis_queue import RedisQueue
from ..queue.redis_streams import RedisStreamsQueue
from ..queue.nats_queue import NatsQueue
from ..queue.sqs_queue import SqsQueue
from ..cluster.node_registry import NodeRegistry
from ..cluster.snapshot_index import SnapshotIndex
from .registry import WorkerRegistry
from .restore_store import RestoreStore


def _deep_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_platform_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).resolve()
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for YAML config files") from exc
    data = yaml.safe_load(raw)
    return data if isinstance(data, dict) else {}


def apply_config_to_env(config: Dict[str, Any]) -> None:
    if not config:
        return
    validate_config(config)
    env_map = {
        "ledger.backend": "KORITH_LEDGER_BACKEND",
        "ledger.sqlite_path": "KORITH_PLATFORM_DB",
        "ledger.postgres_dsn": "KORITH_LEDGER_DSN",
        "artifacts.backend": "KORITH_ARTIFACT_BACKEND",
        "artifacts.local_path": "KORITH_PLATFORM_ARTIFACTS",
        "artifacts.s3_bucket": "KORITH_S3_BUCKET",
        "artifacts.s3_prefix": "KORITH_S3_PREFIX",
        "artifacts.minio_endpoint": "KORITH_MINIO_ENDPOINT",
        "artifacts.minio_access_key": "KORITH_MINIO_ACCESS_KEY",
        "artifacts.minio_secret_key": "KORITH_MINIO_SECRET_KEY",
        "artifacts.minio_bucket": "KORITH_MINIO_BUCKET",
        "artifacts.minio_prefix": "KORITH_MINIO_PREFIX",
        "queue.backend": "KORITH_QUEUE_BACKEND",
        "queue.sqlite_path": "KORITH_QUEUE_DB",
        "queue.redis_url": "KORITH_REDIS_URL",
        "queue.stream_prefix": "KORITH_STREAM_PREFIX",
        "queue.consumer_group": "KORITH_STREAM_CONSUMER_GROUP",
        "queue.nats_url": "KORITH_NATS_URL",
        "queue.sqs_url": "KORITH_SQS_URL",
        "cluster.node_id": "KORITH_NODE_ID",
        "cluster.node_host": "KORITH_NODE_HOST",
        "cluster.node_router_port": "KORITH_NODE_ROUTER_PORT",
        "cluster.registry.sqlite_path": "KORITH_NODE_REGISTRY_DB",
        "cluster.registry.redis_url": "KORITH_NODE_REGISTRY_REDIS_URL",
        "cluster.snapshot_index.redis_url": "KORITH_SNAPSHOT_INDEX_REDIS_URL",
        "cluster.snapshot_bw_mbps": "KORITH_SNAPSHOT_BW_MBPS",
        "cluster.snapshot_rtt_ms": "KORITH_SNAPSHOT_RTT_MS",
        "cluster.transfer_vs_recompute_threshold": "KORITH_TRANSFER_VS_RECOMPUTE_THRESHOLD",
        "registry.sqlite_path": "KORITH_REGISTRY_DB",
        "restore.sqlite_path": "KORITH_RESTORE_DB",
        "runtime.backend_id": "KORITH_DEFAULT_BACKEND_ID",
        "runtime.model_id": "KORITH_DEFAULT_MODEL_ID",
        "runtime.model_path": "KORITH_DEFAULT_MODEL_PATH",
        "runtime.model_endpoint": "KORITH_DEFAULT_MODEL_ENDPOINT",
        "runtime.scheduler.hit_priority": "KORITH_HIT_PRIORITY",
        "runtime.scheduler.miss_priority": "KORITH_MISS_PRIORITY",
        "runtime.scheduler.negative_roi_max": "KORITH_NEGATIVE_ROI_MAX",
        "runtime.scheduler.negative_roi_cooldown_ms": "KORITH_NEGATIVE_ROI_COOLDOWN_MS",
        "runtime.engine.accel_enabled": "KORITH_ACCEL_ENABLED",
        "runtime.engine.cuda_device": "KORITH_CUDA_DEVICE",
        "runtime.engine.kv_layout_version": "KORITH_KV_LAYOUT_VERSION",
        "runtime.engine.fp16": "KORITH_CUDA_FP16",
        "runtime.engine.bf16": "KORITH_CUDA_BF16",
        "runtime.engine.cuda_dtype": "KORITH_CUDA_DTYPE",
        "runtime.spec.enabled": "KORITH_SPEC_ENABLED",
        "runtime.spec.k": "KORITH_SPEC_K",
        "runtime.spec.min_accept": "KORITH_SPEC_MIN_ACCEPT",
        "runtime.spec.disable_after_n": "KORITH_SPEC_DISABLE_AFTER_N",
        "gateway.rate_limit.backend": "KORITH_RATE_LIMIT_BACKEND",
        "gateway.rate_limit.sqlite_path": "KORITH_RATE_LIMIT_DB",
        "gateway.rate_limit.redis_url": "KORITH_REDIS_URL",
        "gateway.rate_limit.burst_factor": "KORITH_RATE_LIMIT_BURST",
        "security.api_key_salt": "KORITH_API_KEY_SALT",
    }
    for key_path, env_name in env_map.items():
        value = _deep_get(config, key_path)
        if value is None:
            continue
        if env_name not in os.environ or not os.environ[env_name]:
            os.environ[env_name] = str(value)


def build_ledger():
    backend = os.environ.get("KORITH_LEDGER_BACKEND", "sqlite")
    if backend == "postgres":
        dsn = os.environ.get("KORITH_LEDGER_DSN", "")
        return PostgresLedgerStore(dsn=dsn)
    db_path = Path(os.environ.get("KORITH_PLATFORM_DB", "./platform_data/ledger.sqlite")).resolve()
    return SQLiteLedgerStore(db_path=db_path)


def build_artifacts():
    backend = os.environ.get("KORITH_ARTIFACT_BACKEND", "local")
    base_dir = Path(os.environ.get("KORITH_PLATFORM_ARTIFACTS", "./platform_data/artifacts")).resolve()
    if backend == "s3":
        bucket = os.environ.get("KORITH_S3_BUCKET", "")
        prefix = os.environ.get("KORITH_S3_PREFIX", "korith")
        return S3ArtifactStore(base_dir=base_dir, bucket=bucket, prefix=prefix)
    if backend == "minio":
        endpoint = os.environ.get("KORITH_MINIO_ENDPOINT", "")
        access_key = os.environ.get("KORITH_MINIO_ACCESS_KEY", "")
        secret_key = os.environ.get("KORITH_MINIO_SECRET_KEY", "")
        bucket = os.environ.get("KORITH_MINIO_BUCKET", "")
        prefix = os.environ.get("KORITH_MINIO_PREFIX", "korith")
        return MinioArtifactStore(
            base_dir=base_dir,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            prefix=prefix,
            secure=os.environ.get("KORITH_MINIO_SECURE", "true").lower() == "true",
        )
    return LocalArtifactStoreAdapter(base_dir=base_dir)


def build_queue():
    backend = os.environ.get("KORITH_QUEUE_BACKEND", "sqlite")
    node_id = os.environ.get("KORITH_NODE_ID", "")
    if backend == "redis":
        return RedisQueue(os.environ.get("KORITH_REDIS_URL", "redis://localhost:6379/0"))
    if backend == "redis_streams":
        return RedisStreamsQueue(
            os.environ.get("KORITH_REDIS_URL", "redis://localhost:6379/0"),
            stream_prefix=os.environ.get("KORITH_STREAM_PREFIX", "korith:jobs"),
            consumer_group=os.environ.get("KORITH_STREAM_CONSUMER_GROUP", "korith"),
            node_id=node_id,
        )
    if backend == "nats":
        return NatsQueue(os.environ.get("KORITH_NATS_URL", "nats://localhost:4222"))
    if backend == "sqs":
        return SqsQueue(os.environ.get("KORITH_SQS_URL", ""))
    if backend == "inproc":
        return InProcQueue()
    db_path = Path(os.environ.get("KORITH_QUEUE_DB", "./platform_data/queue.sqlite")).resolve()
    return SQLiteQueue(db_path=db_path)


def build_registry() -> WorkerRegistry:
    db_path = Path(os.environ.get("KORITH_REGISTRY_DB", "./platform_data/registry.sqlite")).resolve()
    return WorkerRegistry(db_path)


def build_restore_store() -> RestoreStore:
    db_path = Path(os.environ.get("KORITH_RESTORE_DB", "./platform_data/restore.sqlite")).resolve()
    return RestoreStore(db_path)


def build_node_registry() -> NodeRegistry:
    sqlite_path = Path(os.environ.get("KORITH_NODE_REGISTRY_DB", "./platform_data/node_registry.sqlite")).resolve()
    redis_url = os.environ.get("KORITH_NODE_REGISTRY_REDIS_URL", os.environ.get("KORITH_REDIS_URL", ""))
    return NodeRegistry(sqlite_path=sqlite_path, redis_url=redis_url)


def build_snapshot_index(ledger) -> SnapshotIndex:
    redis_url = os.environ.get("KORITH_SNAPSHOT_INDEX_REDIS_URL", os.environ.get("KORITH_REDIS_URL", ""))
    return SnapshotIndex(ledger=ledger, redis_url=redis_url)


def validate_config(config: Dict[str, Any]) -> None:
    if not config:
        return
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    scheduler_cfg = runtime_cfg.get("scheduler", {}) if isinstance(runtime_cfg.get("scheduler"), dict) else {}
    hit_priority = scheduler_cfg.get("hit_priority")
    miss_priority = scheduler_cfg.get("miss_priority")
    if hit_priority is not None and int(hit_priority) < 1:
        raise ValueError("runtime.scheduler.hit_priority must be >= 1")
    if miss_priority is not None and int(miss_priority) < 1:
        raise ValueError("runtime.scheduler.miss_priority must be >= 1")
    engine_cfg = runtime_cfg.get("engine", {}) if isinstance(runtime_cfg.get("engine"), dict) else {}
    spec_cfg = runtime_cfg.get("spec", {}) if isinstance(runtime_cfg.get("spec"), dict) else {}
    if "cuda_device" in engine_cfg and int(engine_cfg["cuda_device"]) < 0:
        raise ValueError("runtime.engine.cuda_device must be >= 0")
    if "k" in spec_cfg and int(spec_cfg["k"]) < 1:
        raise ValueError("runtime.spec.k must be >= 1")
    if "min_accept" in spec_cfg:
        v = float(spec_cfg["min_accept"])
        if v < 0.0 or v > 1.0:
            raise ValueError("runtime.spec.min_accept must be in [0, 1]")
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway"), dict) else {}
    rl_cfg = gateway_cfg.get("rate_limit", {}) if isinstance(gateway_cfg.get("rate_limit"), dict) else {}
    burst = rl_cfg.get("burst_factor")
    if burst is not None and float(burst) < 1.0:
        raise ValueError("gateway.rate_limit.burst_factor must be >= 1.0")
    sec_cfg = config.get("security", {}) if isinstance(config.get("security"), dict) else {}
    salt = sec_cfg.get("api_key_salt")
    if salt is not None:
        if not str(salt).strip() or str(salt).strip() == "korith_default_salt":
            raise ValueError("security.api_key_salt must be set and must not use default value")

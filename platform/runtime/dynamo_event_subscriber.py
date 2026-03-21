"""NATS event subscriber for NVIDIA Dynamo KV Block Manager lifecycle events.

Subscribes to Dynamo's KVBM event plane, aggregates block-level events into
prefix-level state, and triggers AMF persistence for high-value evictions.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .amf_coordinator_client import AmfCoordinatorClient

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DynamoEventConfig:
    """All configuration read from environment variables."""

    nats_url: str = field(default_factory=lambda: os.environ.get("DYNAMO_NATS_URL", "nats://localhost:4222"))
    nats_subject_prefix: str = field(default_factory=lambda: os.environ.get("DYNAMO_NATS_SUBJECT_PREFIX", "dynamo.kv.block"))
    coordinator_url: str = field(default_factory=lambda: os.environ.get("AMF_COORDINATOR_URL", "http://localhost:8500"))
    snapshot_dir: str = field(default_factory=lambda: os.environ.get("AMF_SNAPSHOT_DIR", "/data/amf/snapshots"))
    node_id: str = field(default_factory=lambda: os.environ.get("AMF_NODE_ID", "amf-node-0"))
    min_prefix_tokens: int = field(default_factory=lambda: _env_int("AMF_MIN_PREFIX_TOKENS", 256))
    min_saved_ms_to_persist: float = field(default_factory=lambda: _env_float("AMF_MIN_SAVED_MS_TO_PERSIST", 500.0))
    stats_interval_s: float = field(default_factory=lambda: _env_float("AMF_STATS_INTERVAL_S", 60.0))

    # Storage tier to use for snapshots written by this subscriber.
    default_storage_tier: str = field(default_factory=lambda: os.environ.get("AMF_DEFAULT_STORAGE_TIER", "G3"))


# ---------------------------------------------------------------------------
# KvBlockEvent
# ---------------------------------------------------------------------------

# Storage tier labels used by Dynamo KVBM.
_TIER_LABELS = {
    "G1": "gpu",
    "G2": "cpu",
    "G3": "nvme",
    "G4": "remote",
}


@dataclass
class KvBlockEvent:
    """Parsed representation of a Dynamo KVBM NATS message."""

    event_type: str        # created | evicted | stored | moved
    worker_id: str
    block_id: str
    token_hash: str
    num_tokens: int
    layer_start: int
    layer_end: int
    storage_tier: str      # G1/G2/G3/G4 raw label
    model_id: str
    timestamp_ms: float
    metadata: Dict[str, Any]

    @property
    def storage_tier_label(self) -> str:
        return _TIER_LABELS.get(self.storage_tier, self.storage_tier)

    @classmethod
    def from_nats_payload(cls, subject: str, data: bytes) -> Optional["KvBlockEvent"]:
        """Parse a raw NATS message into a KvBlockEvent.

        Returns None if the message cannot be parsed or is missing required fields.
        """
        parts = subject.rsplit(".", 1)
        if len(parts) < 2:
            logger.debug("Unrecognised NATS subject: %s", subject)
            return None
        event_type = parts[-1]  # created | evicted | stored | moved

        try:
            obj = json.loads(data.decode("utf-8", errors="replace"))
        except Exception as exc:
            logger.debug("Failed to parse NATS payload on %s: %s", subject, exc)
            return None

        if not isinstance(obj, dict):
            return None

        try:
            return cls(
                event_type=str(event_type),
                worker_id=str(obj.get("worker_id", "") or ""),
                block_id=str(obj.get("block_id", "") or ""),
                token_hash=str(obj.get("token_hash", "") or ""),
                num_tokens=int(obj.get("num_tokens", 0) or 0),
                layer_start=int(obj.get("layer_start", 0) or 0),
                layer_end=int(obj.get("layer_end", 0) or 0),
                storage_tier=str(obj.get("storage_tier", "G3") or "G3"),
                model_id=str(obj.get("model_id", "") or ""),
                timestamp_ms=float(obj.get("timestamp_ms", time.time() * 1000) or 0),
                metadata=obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {},
            )
        except (TypeError, ValueError) as exc:
            logger.debug("KvBlockEvent field error on %s: %s", subject, exc)
            return None


# ---------------------------------------------------------------------------
# PrefixState + PrefixAggregator
# ---------------------------------------------------------------------------


@dataclass
class PrefixState:
    """Aggregated state for a single prefix assembled from block events."""

    prefix_id: str           # Derived from worker_id + token_hash of first block.
    worker_id: str
    model_id: str
    tenant_id: str
    total_tokens: int
    block_ids: List[str]
    token_hashes: List[str]
    eviction_count: int
    storage_tier: str
    first_seen_ms: float
    last_seen_ms: float
    estimated_prefill_ms: float


def _estimate_prefill_ms(num_tokens: int) -> float:
    """Rough quadratic approximation for dense transformer prefill cost on H200.

    Calibration points (from Nemotron 120B benchmarks):
      120K tokens → ~243,610 ms
      252K tokens → ~129,690 ms  (hybrid arch — lower due to SSM layers)
      1M   tokens → ~439,358 ms
    We use a simple quadratic fit on the dense-transformer samples.
    """
    t = float(num_tokens)
    # Coefficients fit to (120K, 243610) and (1M, 439358) with a linear component.
    # ms ≈ 0.000001 * t^2 + 0.3 * t
    ms = 1e-6 * t * t + 0.3 * t
    return max(0.0, ms)


def _prefix_id_from_hash(worker_id: str, token_hash: str) -> str:
    key = f"{worker_id}:{token_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class PrefixAggregator:
    """Aggregates Dynamo block-level events into prefix-level state.

    Dynamo operates at block granularity (16-256 tokens/block).  AMF works
    with full-prefix snapshots.  This class maps block events to prefix
    identifiers, accumulates tokens, and detects when a prefix has been
    evicted — at which point it returns the complete PrefixState to the
    caller for AMF persistence.
    """

    def __init__(self, config: DynamoEventConfig) -> None:
        self._config = config
        self._prefixes: Dict[str, PrefixState] = {}

    def handle_event(self, event: KvBlockEvent) -> Optional[PrefixState]:
        """Process a single block event.

        Returns a completed PrefixState when an eviction threshold is reached,
        otherwise returns None.
        """
        if event.event_type == "created":
            self._on_created(event)
        elif event.event_type == "evicted":
            return self._on_evicted(event)
        elif event.event_type in ("stored", "moved"):
            self._on_update(event)
        return None

    def _prefix_key(self, event: KvBlockEvent) -> str:
        return _prefix_id_from_hash(event.worker_id, event.token_hash)

    def _on_created(self, event: KvBlockEvent) -> None:
        key = self._prefix_key(event)
        now_ms = time.time() * 1000
        if key not in self._prefixes:
            self._prefixes[key] = PrefixState(
                prefix_id=key,
                worker_id=event.worker_id,
                model_id=event.model_id,
                tenant_id=event.metadata.get("tenant_id", "default"),
                total_tokens=event.num_tokens,
                block_ids=[event.block_id],
                token_hashes=[event.token_hash],
                eviction_count=0,
                storage_tier=event.storage_tier,
                first_seen_ms=now_ms,
                last_seen_ms=now_ms,
                estimated_prefill_ms=_estimate_prefill_ms(event.num_tokens),
            )
        else:
            state = self._prefixes[key]
            state.total_tokens += event.num_tokens
            state.block_ids.append(event.block_id)
            state.token_hashes.append(event.token_hash)
            state.last_seen_ms = now_ms
            state.estimated_prefill_ms = _estimate_prefill_ms(state.total_tokens)

    def _on_update(self, event: KvBlockEvent) -> None:
        key = self._prefix_key(event)
        state = self._prefixes.get(key)
        if state is None:
            # Block appeared without a prior created event; create stub.
            self._on_created(event)
            return
        state.storage_tier = event.storage_tier
        state.last_seen_ms = time.time() * 1000

    def _on_evicted(self, event: KvBlockEvent) -> Optional[PrefixState]:
        key = self._prefix_key(event)
        state = self._prefixes.get(key)
        if state is None:
            # First we hear of this prefix — create a minimal stub.
            self._on_created(event)
            state = self._prefixes[key]

        state.eviction_count += 1
        state.last_seen_ms = time.time() * 1000

        if state.total_tokens < self._config.min_prefix_tokens:
            logger.debug(
                "Prefix %s skipped: %d tokens < min %d",
                key[:12],
                state.total_tokens,
                self._config.min_prefix_tokens,
            )
            return None

        # Return a copy so we can clean up local state safely.
        completed = PrefixState(
            prefix_id=state.prefix_id,
            worker_id=state.worker_id,
            model_id=state.model_id,
            tenant_id=state.tenant_id,
            total_tokens=state.total_tokens,
            block_ids=list(state.block_ids),
            token_hashes=list(state.token_hashes),
            eviction_count=state.eviction_count,
            storage_tier=state.storage_tier,
            first_seen_ms=state.first_seen_ms,
            last_seen_ms=state.last_seen_ms,
            estimated_prefill_ms=state.estimated_prefill_ms,
        )
        # Remove from active tracking after eviction.
        self._prefixes.pop(key, None)
        return completed

    @property
    def active_prefix_count(self) -> int:
        return len(self._prefixes)


# ---------------------------------------------------------------------------
# AmfPersistBridge
# ---------------------------------------------------------------------------


class AmfPersistBridge:
    """Decides whether to persist a PrefixState to AMF and executes the persist.

    Unlike LMCache which caches everything, AMF only persists prefixes where
    estimated_prefill_ms exceeds the configured minimum threshold (ROI gate).
    """

    def __init__(self, config: DynamoEventConfig, coordinator_client: AmfCoordinatorClient) -> None:
        self._config = config
        self._client = coordinator_client
        self._persist_count = 0
        self._reject_count = 0
        self._total_tokens_persisted = 0

    def maybe_persist(self, state: PrefixState) -> bool:
        """Evaluate ROI gate and persist if worthy.

        Returns True if the entry was registered with the coordinator.
        """
        if state.estimated_prefill_ms < self._config.min_saved_ms_to_persist:
            self._reject_count += 1
            logger.debug(
                "Prefix %s rejected: estimated_prefill_ms=%.1f < threshold=%.1f",
                state.prefix_id[:12],
                state.estimated_prefill_ms,
                self._config.min_saved_ms_to_persist,
            )
            return False

        snapshot_path = os.path.join(
            self._config.snapshot_dir,
            state.tenant_id,
            state.model_id,
            f"{state.prefix_id}.amf",
        )

        metadata: Dict[str, Any] = {
            "model_id": state.model_id,
            "total_tokens": state.total_tokens,
            "estimated_prefill_ms": state.estimated_prefill_ms,
            "snapshot_path": snapshot_path,
            "storage_tier": state.storage_tier,
            "eviction_count": state.eviction_count,
            "block_count": len(state.block_ids),
            "source": "dynamo_event_subscriber",
        }

        ok = self._client.register(
            prefix_hash=state.prefix_id,
            tenant_id=state.tenant_id,
            node_id=self._config.node_id,
            worker_id=state.worker_id,
            metadata=metadata,
        )

        if ok:
            self._persist_count += 1
            self._total_tokens_persisted += state.total_tokens
            logger.info(
                "AMF persist: prefix=%s tokens=%d prefill_ms=%.0f tier=%s",
                state.prefix_id[:12],
                state.total_tokens,
                state.estimated_prefill_ms,
                state.storage_tier,
            )
        else:
            logger.warning("AMF coordinator register failed for prefix %s", state.prefix_id[:12])

        return ok

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "persist_count": self._persist_count,
            "reject_count": self._reject_count,
            "total_tokens_persisted": self._total_tokens_persisted,
        }


# ---------------------------------------------------------------------------
# DynamoEventSubscriber
# ---------------------------------------------------------------------------


class DynamoEventSubscriber:
    """Connects to Dynamo's NATS event plane and drives AMF persistence.

    Usage::

        config = DynamoEventConfig()
        subscriber = DynamoEventSubscriber(config)
        asyncio.run(subscriber.start())
    """

    _SUBJECTS_SUFFIXES = ("created", "evicted", "stored", "moved")

    def __init__(self, config: Optional[DynamoEventConfig] = None) -> None:
        self._config = config or DynamoEventConfig()
        self._coordinator = AmfCoordinatorClient(self._config.coordinator_url)
        self._aggregator = PrefixAggregator(self._config)
        self._bridge = AmfPersistBridge(self._config, self._coordinator)
        self._msg_count = 0
        self._error_count = 0
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and begin processing events until stopped."""
        try:
            import nats  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "nats-py is required: pip install nats-py"
            ) from exc

        logger.info("DynamoEventSubscriber connecting to NATS at %s", self._config.nats_url)
        nc = await nats.connect(self._config.nats_url)
        self._running = True

        subjects = [
            f"{self._config.nats_subject_prefix}.{suffix}"
            for suffix in self._SUBJECTS_SUFFIXES
        ]

        async def _handler(msg: Any) -> None:
            self._msg_count += 1
            try:
                event = KvBlockEvent.from_nats_payload(msg.subject, msg.data)
                if event is None:
                    return
                state = self._aggregator.handle_event(event)
                if state is not None:
                    self._bridge.maybe_persist(state)
            except Exception as exc:
                self._error_count += 1
                logger.exception("Error processing NATS message on %s: %s", msg.subject, exc)

        subs = []
        for subject in subjects:
            sub = await nc.subscribe(subject, cb=_handler)
            subs.append(sub)
            logger.info("Subscribed to NATS subject: %s", subject)

        stats_task = asyncio.ensure_future(self._stats_loop())

        logger.info("DynamoEventSubscriber ready — waiting for events")
        try:
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            stats_task.cancel()
            for sub in subs:
                await sub.unsubscribe()
            await nc.close()
            logger.info("DynamoEventSubscriber stopped")

    def stop(self) -> None:
        self._running = False

    async def _stats_loop(self) -> None:
        interval = max(5.0, float(self._config.stats_interval_s))
        while True:
            await asyncio.sleep(interval)
            bridge_stats = self._bridge.stats
            logger.info(
                "DynamoEventSubscriber stats: messages=%d errors=%d active_prefixes=%d "
                "persisted=%d rejected=%d tokens_persisted=%d",
                self._msg_count,
                self._error_count,
                self._aggregator.active_prefix_count,
                bridge_stats["persist_count"],
                bridge_stats["reject_count"],
                bridge_stats["total_tokens_persisted"],
            )

    @property
    def bridge(self) -> AmfPersistBridge:
        return self._bridge

    @property
    def aggregator(self) -> PrefixAggregator:
        return self._aggregator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = DynamoEventConfig()
    logger.info(
        "Starting DynamoEventSubscriber: nats=%s prefix=%s node=%s min_tokens=%d min_saved_ms=%.0f",
        config.nats_url,
        config.nats_subject_prefix,
        config.node_id,
        config.min_prefix_tokens,
        config.min_saved_ms_to_persist,
    )
    subscriber = DynamoEventSubscriber(config)
    asyncio.run(subscriber.start())


if __name__ == "__main__":
    main()

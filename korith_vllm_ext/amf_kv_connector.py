"""amf_kv_connector.py — KVConnector plugin for vLLM v0.19.

Implements KVConnectorBase_V1 to wire AMF's compressed VRAM pool into
vLLM's KV transfer system. On cold miss, vLLM prefills normally and
save_kv_layer() captures each layer's KV. On warm hit, vLLM skips
prefill and start_load_kv() decompresses KV from the pool directly
into vLLM's paged buffer.

The connector runs as TWO independent instances:
  - Scheduler side: get_num_new_matched_tokens, build_connector_meta
  - Worker side: start_load_kv, save_kv_layer, wait_for_save

Both maintain their own AMF pool. The scheduler pool is used only for
lookup (has/hasn't); the worker pool holds actual compressed KV data.

Usage:
    LLM(
        model="...",
        kv_transfer_config={
            "kv_connector": "AMFKVConnector",
            "kv_connector_module_path": "korith_vllm_ext.amf_kv_connector",
            "kv_role": "kv_both",
            "kv_buffer_size": 100000000,
            "kv_connector_extra_config": {
                "num_layers": 28,
                "num_kv_heads": 2,
                "head_dim": 64,
                "bytes_per_layer": 8388608,
                "default_format": "fp8_e4m3",
                "min_prefix_tokens": 64,
            },
        },
    )
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger("amf_kv_connector")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _prefix_hash(token_ids: list[int] | tuple[int, ...]) -> str:
    """Content-addressed hash over prompt token IDs."""
    h = hashlib.sha256()
    for t in token_ids:
        h.update(int(t & 0xFFFFFFFF).to_bytes(4, "little"))
    return h.hexdigest()[:16]


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Round down to the nearest multiple of block_size."""
    return (num_tokens - 1) // block_size * block_size


# ── Connector metadata ───────────────────────────────────────────────────────


@dataclass
class AMFReqMeta:
    """Per-request metadata passed from scheduler to worker."""
    request_id: str
    prefix_hash: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    num_tokens: int

    @staticmethod
    def make(
        request_id: str,
        prefix_hash: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> "AMFReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids[:valid_num_tokens])
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape(1, block_size)
            + block_ids_tensor.reshape(num_blocks, 1) * block_size
        ).flatten()[:valid_num_tokens]
        return AMFReqMeta(
            request_id=request_id,
            prefix_hash=prefix_hash,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            num_tokens=valid_num_tokens,
        )


@dataclass
class AMFConnectorMetadata(KVConnectorMetadata):
    requests: list[AMFReqMeta] = field(default_factory=list)


# ── The connector ─────────────────────────────────────────────────────────────


class AMFKVConnector(KVConnectorBase_V1):
    """AMF compressed VRAM pool as a vLLM KVConnector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}

        # Read AMF config from extra_config.
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._num_layers = int(extra.get("num_layers", 28))
        self._num_kv_heads = int(extra.get("num_kv_heads", 2))
        self._head_dim = int(extra.get("head_dim", 64))
        self._bytes_per_layer = int(extra.get("bytes_per_layer", 1 << 23))
        self._default_format = str(extra.get("default_format", "fp8_e4m3"))
        self._min_prefix_tokens = int(extra.get("min_prefix_tokens", 64))

        # Pool is created lazily so import errors don't crash the scheduler
        # when it doesn't have GPU access.
        self._pool = None
        self._pool_init_done = False

        # Per-step save buffer: prefix_hash -> {layer_name: kv_tensor}
        self._save_buffer: dict[str, dict[str, torch.Tensor]] = {}
        # Track prefix hashes that the scheduler knows about (for fast lookup).
        self._known_hashes: set[str] = set()
        # Cache for layer name -> index mapping.
        self._layer_idx_cache: dict[str, int] = {}

        logger.info(
            "[AMF_CONNECTOR] init role=%s layers=%d heads=%d head_dim=%d "
            "bytes_per_layer=%d format=%s block_size=%d",
            role.name, self._num_layers, self._num_kv_heads,
            self._head_dim, self._bytes_per_layer, self._default_format,
            self._block_size,
        )

    # ── Lazy pool init ─────────────────────────────────────────────────────

    def _ensure_pool(self):
        if self._pool_init_done:
            return self._pool
        self._pool_init_done = True
        try:
            from korith_vllm_ext.compressed_vram_pool import CompressedVRAMPool
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pool = CompressedVRAMPool(
                num_layers=self._num_layers,
                bytes_per_layer=self._bytes_per_layer,
                block_bytes=1 << 17,  # 128 KB
                default_format=self._default_format,
                device=device,
            )
            logger.info(
                "[AMF_CONNECTOR] Pool created: %d layers x %d bytes = %.1f MB on %s",
                self._num_layers, self._bytes_per_layer,
                self._num_layers * self._bytes_per_layer / (1 << 20), device,
            )
        except Exception as exc:
            logger.warning("[AMF_CONNECTOR] Pool init failed: %s", exc)
            self._pool = None
        return self._pool

    # ======================================================================
    # Scheduler-side methods
    # ======================================================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check if the AMF pool has KV for this request's prefix."""
        token_ids = list(request.prompt_token_ids or [])
        if len(token_ids) < self._min_prefix_tokens:
            return 0, False

        # Hash the prompt (minus the last token, following ExampleConnector).
        num_tokens_to_check = align_to_block_size(
            len(token_ids) - 1, self._block_size
        )
        if num_tokens_to_check <= 0:
            return 0, False

        prefix_hash = _prefix_hash(token_ids[:num_tokens_to_check])

        # Check if we know about this hash (either from a previous save
        # or from the pool itself).
        pool = self._ensure_pool()
        has_hit = prefix_hash in self._known_hashes
        if not has_hit and pool is not None:
            has_hit = prefix_hash in pool

        if not has_hit:
            return 0, False

        matched = num_tokens_to_check - num_computed_tokens
        if matched <= 0:
            return 0, False

        logger.info(
            "[AMF_CONNECTOR] HIT prefix=%s matched=%d computed=%d total=%d",
            prefix_hash[:12], matched, num_computed_tokens, num_tokens_to_check,
        )
        return matched, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        meta = AMFConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            num_tokens = align_to_block_size(len(token_ids) - 1, self._block_size)
            prefix_hash = _prefix_hash(token_ids[:num_tokens]) if num_tokens > 0 else ""

            if new_req.req_id in self._requests_need_load:
                # This request needs KV loaded from the pool.
                meta.requests.append(AMFReqMeta.make(
                    request_id=new_req.req_id,
                    prefix_hash=prefix_hash,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                ))
                total_need_load += 1
            else:
                # Cold miss — mark for save after prefill.
                meta.requests.append(AMFReqMeta.make(
                    request_id=new_req.req_id,
                    prefix_hash=prefix_hash,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=True,
                ))

        # Handle resumed requests that need loading.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed = req_id in cached_reqs.resumed_req_ids
            if not resumed or req_id not in self._requests_need_load:
                continue
            request = self._requests_need_load[req_id]
            num_computed = cached_reqs.num_computed_tokens[i]
            num_new = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = num_computed + num_new
            token_ids = list(request.all_token_ids[:total_tokens])
            num_tokens = align_to_block_size(len(token_ids) - 1, self._block_size)
            prefix_hash = _prefix_hash(token_ids[:num_tokens]) if num_tokens > 0 else ""
            new_block_ids = cached_reqs.new_block_ids[i]
            assert new_block_ids is not None
            meta.requests.append(AMFReqMeta.make(
                request_id=req_id,
                prefix_hash=prefix_hash,
                token_ids=token_ids,
                block_ids=new_block_ids[0],
                block_size=self._block_size,
                is_store=False,
            ))
            total_need_load += 1

        assert total_need_load == len(self._requests_need_load), (
            f"Expected {len(self._requests_need_load)} loads, "
            f"scheduled {total_need_load}"
        )
        self._requests_need_load.clear()
        return meta

    # ======================================================================
    # Worker-side methods
    # ======================================================================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Decompress KV from AMF pool into vLLM's paged buffer."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, AMFConnectorMetadata)

        pool = self._ensure_pool()
        if pool is None:
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        for req in metadata.requests:
            if req.is_store:
                continue

            entry = pool.get(req.prefix_hash)
            if entry is None:
                logger.warning(
                    "[AMF_CONNECTOR] LOAD MISS prefix=%s (pool has %d entries)",
                    req.prefix_hash[:12], pool.num_prefixes(),
                )
                continue

            logger.info(
                "[AMF_CONNECTOR] Loading %d tokens for prefix=%s",
                req.num_tokens, req.prefix_hash[:12],
            )

            from korith_vllm_ext.codecs import get_codec

            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue

                # Find which layer index this corresponds to.
                layer_idx = self._layer_name_to_idx(layer_name)
                if layer_idx is None or layer_idx >= len(entry.blobs):
                    continue

                blob = entry.blobs[layer_idx]
                codec = get_codec(blob.format)
                kv_data = codec.decompress_to(blob, target_dtype=kv_cache_layer.dtype)
                # kv_data shape: [2, num_tokens, num_heads, head_dim]
                # Flatten to [2, num_tokens, head_dim * num_heads] for slot indexing.
                if kv_data.dim() == 4:
                    two, n_tok, n_heads, hdim = kv_data.shape
                    kv_data = kv_data.reshape(two, n_tok, n_heads * hdim)

                slot_mapping = req.slot_mapping.to(kv_cache_layer.device)
                valid = min(kv_data.shape[1], slot_mapping.shape[0])
                kv_slice = kv_data[:, :valid, ...].to(kv_cache_layer.device)
                slots = slot_mapping[:valid]

                # Inject into the paged KV cache.
                # Layout: [2, num_pages, page_size, ...] → reshape to [2, total_slots, ...]
                shape = kv_cache_layer.shape
                num_pages = shape[1]
                page_size = shape[2]
                flat_cache = kv_cache_layer.reshape(2, num_pages * page_size, -1)
                flat_cache[:, slots, ...] = kv_slice

            logger.info(
                "[AMF_CONNECTOR] Loaded prefix=%s into paged buffer",
                req.prefix_hash[:12],
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No async loading — start_load_kv is synchronous."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Extract KV from the paged buffer for cold-prefill requests."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, AMFConnectorMetadata)

        for req in metadata.requests:
            if not req.is_store:
                continue

            slot_mapping = req.slot_mapping.to(kv_layer.device)
            valid = min(slot_mapping.shape[0], req.num_tokens)
            slots = slot_mapping[:valid]

            # Extract KV from the paged buffer.
            # kv_layer shape: [2, num_pages, page_size, num_kv_heads, head_dim]
            shape = kv_layer.shape
            if len(shape) == 5 and shape[0] == 2:
                num_pages = shape[1]
                page_size = shape[2]
                flat = kv_layer.reshape(2, num_pages * page_size, shape[3], shape[4])
                kv_data = flat[:, slots, :, :].detach().clone()
            elif len(shape) == 4:
                # [2, total_slots, num_kv_heads, head_dim] already flat
                kv_data = kv_layer[:, slots, :, :].detach().clone()
            elif len(shape) == 3:
                # [2, total_slots, hidden_dim]
                kv_data = kv_layer[:, slots, :].detach().clone()
                # Reshape to [2, T, num_kv_heads, head_dim]
                kv_data = kv_data.reshape(2, valid, self._num_kv_heads, self._head_dim)
            else:
                logger.warning(
                    "[AMF_CONNECTOR] Unexpected kv_layer shape %s for %s",
                    shape, layer_name,
                )
                continue

            # Buffer this layer's KV for wait_for_save.
            key = req.prefix_hash
            if key not in self._save_buffer:
                self._save_buffer[key] = {}
            self._save_buffer[key][layer_name] = kv_data

    def wait_for_save(self) -> None:
        """Compress buffered layers and store in the AMF pool."""
        pool = self._ensure_pool()
        if pool is None or not self._save_buffer:
            self._save_buffer.clear()
            return

        for prefix_hash, layers_dict in self._save_buffer.items():
            if not layers_dict:
                continue

            # Sort layers by index to get correct ordering.
            sorted_names = sorted(
                layers_dict.keys(),
                key=lambda n: self._layer_name_to_idx(n) or 0,
            )

            # Stack into [num_layers, 2, T, H, D].
            layer_tensors = []
            for name in sorted_names:
                kv = layers_dict[name]  # [2, T, H, D]
                layer_tensors.append(kv)

            if not layer_tensors:
                continue

            stacked = torch.stack(layer_tensors, dim=0)  # [L, 2, T, H, D]
            logger.info(
                "[AMF_CONNECTOR] Saving prefix=%s shape=%s (%d layers)",
                prefix_hash[:12], tuple(stacked.shape), len(layer_tensors),
            )

            ok = pool.put_from_raw(
                prefix_hash,
                stacked,
                format=self._default_format,
            )
            if ok:
                self._known_hashes.add(prefix_hash)
                logger.info(
                    "[AMF_CONNECTOR] Saved prefix=%s (%d layers, pool=%d entries)",
                    prefix_hash[:12], len(layer_tensors), pool.num_prefixes(),
                )
            else:
                logger.warning(
                    "[AMF_CONNECTOR] Save FAILED for prefix=%s",
                    prefix_hash[:12],
                )

        self._save_buffer.clear()

    # ── Helper ─────────────────────────────────────────────────────────────

    def _layer_name_to_idx(self, layer_name: str) -> int | None:
        """Extract the layer index from a vLLM layer name.

        Examples:
            "model.layers.0.self_attn" -> 0
            "model.layers.27.self_attn" -> 27
        """
        if layer_name in self._layer_idx_cache:
            return self._layer_idx_cache[layer_name]
        parts = layer_name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    idx = int(parts[i + 1])
                    self._layer_idx_cache[layer_name] = idx
                    return idx
                except ValueError:
                    pass
        return None

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """AMF needs piecewise CUDA graphs since save_kv_layer runs Python."""
        return True

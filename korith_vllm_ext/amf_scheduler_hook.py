"""amf_scheduler_hook.py — AMF prefill-skip hook for vLLM's scheduling loop.

Hooks into KorithDecodeScheduler to check the AMF store before a sequence is
sent to prefill workers.  If a snapshot exists, KV cache is restored directly
and the sequence jumps straight to the decode queue.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .amf_kv_manager import AmfKvManager

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return (v in _TRUTHY) if v else default


class AmfSchedulerHook:
    """Hooks into vLLM's scheduling loop to check AMF before prefill.

    Usage in KorithDecodeScheduler:
        self._amf_hook = AmfSchedulerHook(kv_manager)

        # In _schedule_prefills():
        if self._amf_hook.before_prefill(seq_group, block_table):
            continue  # skip prefill for this group

        # After prefill completes:
        self._amf_hook.after_prefill(seq_group, block_table)
    """

    def __init__(self, kv_manager: AmfKvManager) -> None:
        self._kv_manager = kv_manager
        self._enabled = _env_truthy("KORITH_ENABLE_AMF", default=True)
        self._hits   = 0
        self._misses = 0
        self._saves  = 0
        self._total_restore_ms = 0.0
        self._min_tokens = int(os.environ.get("KORITH_AMF_MIN_TOKENS", "64") or 64)

    # ── Before-prefill hook ───────────────────────────────────────────────────

    def before_prefill(
        self,
        seq_group: Any,
        block_table: List[int],
    ) -> bool:
        """Check AMF before a sequence is sent to prefill workers.

        Returns True if the prefill should be skipped (AMF hit + restore OK).
        The caller is responsible for moving the sequence to the decode queue.
        """
        if not self._enabled:
            return False

        prompt_tokens = self._extract_prompt_tokens(seq_group)
        if not prompt_tokens or len(prompt_tokens) < self._min_tokens:
            return False

        if not self._kv_manager.has_snapshot(prompt_tokens):
            self._misses += 1
            logger.debug("[AMF_MISS] tokens=%d", len(prompt_tokens))
            return False

        t0 = time.monotonic()
        n_restored = self._kv_manager.restore_kv_state(prompt_tokens, block_table)
        restore_ms = (time.monotonic() - t0) * 1000.0

        if n_restored <= 0:
            self._misses += 1
            logger.warning(
                "[AMF_RESTORE_FAIL] tokens=%d restore_ms=%.1f",
                len(prompt_tokens),
                restore_ms,
            )
            return False

        self._hits += 1
        self._total_restore_ms += restore_ms

        # Update sequence's computed token count so vLLM knows it's prefilled.
        self._mark_prefill_complete(seq_group, n_restored)

        logger.info(
            "[AMF_HIT] prefix_tokens=%d restore_ms=%.1f",
            n_restored,
            restore_ms,
        )
        print(
            f"[AMF_SKIP] prompt_tokens={len(prompt_tokens)} "
            f"skipped_tokens={n_restored} "
            f"skip_ratio={n_restored/max(1,len(prompt_tokens)):.3f} "
            f"restore_ms={restore_ms:.2f}",
            flush=True,
        )
        return True

    # ── After-prefill hook ────────────────────────────────────────────────────

    def after_prefill(
        self,
        seq_group: Any,
        block_table: List[int],
        *,
        saved_ms: float = 0.0,
    ) -> None:
        """Called after prefill completes.  Saves KV snapshot if ROI admits it."""
        if not self._enabled:
            return

        prompt_tokens = self._extract_prompt_tokens(seq_group)
        if not prompt_tokens or len(prompt_tokens) < self._min_tokens:
            return

        # Simple ROI gate: only save if saved_ms > 0 or not set (cold run).
        ok = self._kv_manager.save_kv_state(
            prompt_tokens,
            block_table,
            saved_ms=saved_ms,
        )
        if ok:
            self._saves += 1
            logger.info(
                "[AMF_STATS] saved tokens=%d", len(prompt_tokens)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_prompt_tokens(seq_group: Any) -> Optional[List[int]]:
        """Extract prompt token IDs from a vLLM SequenceGroup."""
        try:
            # vLLM SequenceGroup API.
            seqs = seq_group.get_seqs()
            if not seqs:
                return None
            seq_data = seqs[0].data
            return list(seq_data.prompt_token_ids)
        except AttributeError:
            pass
        try:
            return list(seq_group.prompt_token_ids)
        except AttributeError:
            return None

    @staticmethod
    def _mark_prefill_complete(seq_group: Any, n_tokens: int) -> None:
        """Tell vLLM that this sequence has already been prefilled."""
        try:
            for seq in seq_group.get_seqs():
                seq.data.update_num_computed_tokens(n_tokens)
        except AttributeError:
            logger.warning("[AMF_VLLM] could not mark sequence as prefill-complete")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "hits":             self._hits,
            "misses":           self._misses,
            "saves":            self._saves,
            "avg_restore_ms":   self._total_restore_ms / max(1, self._hits),
            "total_restore_ms": self._total_restore_ms,
        }

    def log_summary(self) -> None:
        s = self.stats()
        total = s["hits"] + s["misses"]
        hit_rate = s["hits"] / max(1, total)
        print(
            f"[KORITH_RUN_SUMMARY] amf_hits={s['hits']} amf_misses={s['misses']} "
            f"amf_saves={s['saves']} hit_rate={hit_rate:.3f} "
            f"avg_restore_ms={s['avg_restore_ms']:.1f}",
            flush=True,
        )

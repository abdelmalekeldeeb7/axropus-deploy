from __future__ import annotations

from typing import Any, Dict, Optional


class SpecDecodeController:
    """Python-side speculative decode controller.

    Manages dynamic k (depth) adjustment based on acceptance rate and exposes
    metrics: acceptance_rate, tokens_proposed, tokens_accepted, decode_ms_saved,
    spec_overhead_ms.

    Thresholds (matching C++ thermo_controller):
        accept > 0.85  → increase k  (up to max_k=12)
        accept < 0.65  → decrease k  (down to min_k=2)
    """

    # Dynamic k bounds.
    MIN_K: int = 2
    MAX_K: int = 12
    DEFAULT_K: int = 4

    # Acceptance rate thresholds for k adjustment.
    ACCEPT_INCREASE_THRESHOLD: float = 0.85
    ACCEPT_DECREASE_THRESHOLD: float = 0.65

    # EMA smoothing factor for acceptance rate.
    ACCEPT_EMA_ALPHA: float = 0.2

    def __init__(self) -> None:
        self.enabled: bool = False
        self.current_k: int = self.DEFAULT_K

        # Cumulative metrics.
        self.tokens_proposed: int = 0
        self.tokens_accepted: int = 0
        self.accept_ema: float = 0.0
        self.decode_ms_saved: float = 0.0
        self.spec_overhead_ms: float = 0.0

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Gate: decide whether speculation should run this step.
    # ------------------------------------------------------------------

    def maybe_run(
        self,
        capabilities: Dict[str, bool],
        policy: Dict[str, bool],
    ) -> Dict[str, Any]:
        if not policy.get("allow_spec", False):
            return {"enabled": 0.0, "reason": "policy_disabled", "k": 0}
        if not capabilities.get("VERIFY_TOKENS") and not capabilities.get("LOGITS_ACCESS"):
            return {"enabled": 0.0, "reason": "capability_missing", "k": 0}

        self.enabled = True
        return {
            "enabled": 1.0,
            "reason": "ok",
            "k": self.current_k,
        }

    # ------------------------------------------------------------------
    # Update: feed back per-step results from the C++ engine.
    # ------------------------------------------------------------------

    def update(
        self,
        proposed: int,
        accepted: int,
        spec_elapsed_ns: int = 0,
        baseline_tps: float = 0.0,
    ) -> None:
        """Update controller state after a speculative decode step."""
        self.tokens_proposed += proposed
        self.tokens_accepted += accepted

        if proposed > 0:
            step_accept = accepted / proposed
            if not self._initialized:
                self.accept_ema = step_accept
                self._initialized = True
            else:
                self.accept_ema += self.ACCEPT_EMA_ALPHA * (step_accept - self.accept_ema)

            # Dynamic k adjustment.
            if self.accept_ema > self.ACCEPT_INCREASE_THRESHOLD:
                self.current_k = min(self.MAX_K, self.current_k + 1)
            elif self.accept_ema < self.ACCEPT_DECREASE_THRESHOLD:
                self.current_k = max(self.MIN_K, self.current_k - 1)

        # Overhead tracking.
        step_overhead_ms = spec_elapsed_ns * 1e-6
        self.spec_overhead_ms += step_overhead_ms

        # Decode savings estimation.
        if accepted > 1 and baseline_tps > 1e-6:
            baseline_ms_per_token = 1000.0 / baseline_tps
            tokens_saved = accepted - 1
            step_saved = tokens_saved * baseline_ms_per_token - step_overhead_ms
            self.decode_ms_saved += max(0.0, step_saved)

    # ------------------------------------------------------------------
    # Metrics snapshot for telemetry / governance.
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        acceptance_rate = (
            self.tokens_accepted / self.tokens_proposed
            if self.tokens_proposed > 0
            else 0.0
        )
        return {
            "enabled": self.enabled,
            "k": self.current_k,
            "acceptance_rate": round(acceptance_rate, 4),
            "accept_ema": round(self.accept_ema, 4),
            "tokens_proposed": self.tokens_proposed,
            "tokens_accepted": self.tokens_accepted,
            "decode_ms_saved": round(self.decode_ms_saved, 2),
            "spec_overhead_ms": round(self.spec_overhead_ms, 2),
        }

    def reset(self) -> None:
        """Reset all state (e.g. between requests)."""
        self.enabled = False
        self.current_k = self.DEFAULT_K
        self.tokens_proposed = 0
        self.tokens_accepted = 0
        self.accept_ema = 0.0
        self.decode_ms_saved = 0.0
        self.spec_overhead_ms = 0.0
        self._initialized = False

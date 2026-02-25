from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Capabilities:
    kv_replay: bool = False
    deterministic_seeding: bool = False
    streaming: bool = False
    batch_prefill: bool = False
    logits_access: bool = False
    verify_tokens: bool = False
    draft_supported: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "KV_REPLAY": self.kv_replay,
            "DETERMINISTIC_SEEDING": self.deterministic_seeding,
            "STREAMING": self.streaming,
            "BATCH_PREFILL": self.batch_prefill,
            "LOGITS_ACCESS": self.logits_access,
            "VERIFY_TOKENS": self.verify_tokens,
            "DRAFT_SUPPORTED": self.draft_supported,
        }


class BackendAdapter:
    backend_id = "base"
    backend_version = "v0"

    def get_fingerprint(self) -> Dict[str, str]:
        raise NotImplementedError

    def get_capabilities(self) -> Capabilities:
        raise NotImplementedError

    def tokenize(self, prompt: str) -> int:
        raise NotImplementedError

    def run_baseline(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
    ) -> Dict:
        raise NotImplementedError

    def run_accel(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
        engine_cfg: Optional[Dict] = None,
    ) -> Dict:
        return self.run_baseline(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=policy,
            artifacts=artifacts,
            mf_snapshot_in=mf_snapshot_in,
        )

    def run_speculative(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
        spec_cfg: Optional[Dict] = None,
        engine_cfg: Optional[Dict] = None,
    ) -> Dict:
        return self.run_accel(
            prompt=prompt,
            max_tokens=max_tokens,
            deterministic_cfg=deterministic_cfg,
            policy=policy,
            artifacts=artifacts,
            mf_snapshot_in=mf_snapshot_in,
            engine_cfg=engine_cfg,
        )

    def run_draft(
        self,
        prompt: str,
        deterministic_cfg: Dict,
        max_tokens: int,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        raise NotImplementedError

    def verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[int],
        deterministic_cfg: Dict,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        raise NotImplementedError

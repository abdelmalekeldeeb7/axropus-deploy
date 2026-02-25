from __future__ import annotations

from typing import Dict

from .base import Capabilities
from .korith_local import Tier1LocalKorithAdapter


class KorithCudaAdapter(Tier1LocalKorithAdapter):
    backend_id = "korith_cuda"
    backend_version = "korith_cuda_v1"

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path=model_path, backend_id=self.backend_id, backend_version=self.backend_version)

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            kv_replay=True,
            deterministic_seeding=True,
            streaming=False,
            batch_prefill=True,
            logits_access=True,
            verify_tokens=True,
            draft_supported=True,
        )

    def get_fingerprint(self) -> Dict[str, str]:
        fp = super().get_fingerprint()
        fp["backend_id"] = self.backend_id
        fp["backend_version"] = self.backend_version
        return fp


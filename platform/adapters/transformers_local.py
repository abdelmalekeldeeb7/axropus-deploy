from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import BackendAdapter, Capabilities


class HFTransformersAdapter(BackendAdapter):
    backend_id = "hf_transformers"
    backend_version = "v1"

    def __init__(self, model_id: str, model_path: Optional[str] = None, device: str = "cpu") -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("transformers/torch not installed") from exc
        self._torch = torch
        self._model_id = model_id
        self._model_path = model_path
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_path or model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_path or model_id)
        self._model.to(device)
        self._model.eval()

    def get_fingerprint(self) -> Dict[str, str]:
        base = self._model_path or self._model_id
        model_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
        tokenizer_hash = hashlib.sha256((base + ":tokenizer").encode("utf-8")).hexdigest()
        return {
            "model_hash": model_hash,
            "tokenizer_hash": tokenizer_hash,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
        }

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            kv_replay=False,
            deterministic_seeding=True,
            streaming=False,
            batch_prefill=False,
            logits_access=True,
            verify_tokens=False,
            draft_supported=False,
        )

    def tokenize(self, prompt: str) -> int:
        return len(self._tokenizer.encode(prompt))

    def run_baseline(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
    ) -> Dict:
        start = time.time()
        seed = int(deterministic_cfg.get("seed", 0))
        self._torch.manual_seed(seed)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prefill_done = time.time()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        end = time.time()
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        Path(artifacts["output"]).write_text(text, encoding="utf-8")
        perf = {
            "prefill_ms": (prefill_done - start) * 1000.0,
            "decode_ms": (end - prefill_done) * 1000.0,
            "total_ms": (end - start) * 1000.0,
            "tokens_out": int(len(outputs[0])),
            "avg_tps": 0.0,
        }
        return {
            "exit_code": 0,
            "output_text": text,
            "total_ms": perf["total_ms"],
            "engine_metrics": {
                "amf": {"supported": False, "decision": "unavailable"},
                "mf": {"supported": False},
                "perf": perf,
                "engine": {"mode": "baseline", "accel_enabled": False},
                "spec": {"enabled": False, "reason": "backend_no_spec"},
                "health": {},
            },
        }

    def run_draft(
        self,
        prompt: str,
        deterministic_cfg: Dict,
        max_tokens: int,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        _ = deterministic_cfg
        n = max(1, min(int((spec_cfg or {}).get("draft_max_tokens", max_tokens)), max_tokens))
        toks = self._tokenizer.encode(prompt)[:n]
        return {"draft_tokens": [int(t) for t in toks], "state": {"device": self._device}}

    def verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[int],
        deterministic_cfg: Dict,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        _ = (prompt, deterministic_cfg, spec_cfg)
        return {"accepted_count": int(len(draft_tokens)), "verified_logits": None}

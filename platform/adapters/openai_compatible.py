from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

from .base import BackendAdapter, Capabilities


class Tier2OpenAICompatibleAdapter(BackendAdapter):
    backend_id = "openai_compatible"
    backend_version = "v1"

    def __init__(self, endpoint: str, model_id: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id

    def get_fingerprint(self) -> Dict[str, str]:
        return {
            "model_hash": f"remote:{self.model_id}",
            "tokenizer_hash": "unknown",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
        }

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            kv_replay=False,
            deterministic_seeding=True,
            streaming=False,
            batch_prefill=False,
            verify_tokens=False,
            draft_supported=False,
        )

    def tokenize(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    def _resolve_url(self) -> str:
        url = self.endpoint
        if not url.endswith("/v1/completions") and not url.endswith("/v1/chat/completions"):
            if url.endswith("/v1"):
                url = url + "/completions"
            else:
                url = url + "/v1/completions"
        return url

    def run_baseline(
        self,
        prompt: str,
        max_tokens: int,
        deterministic_cfg: Dict,
        policy: Dict,
        artifacts: Dict[str, str],
        mf_snapshot_in: Optional[str],
    ) -> Dict:
        url = self._resolve_url()
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(deterministic_cfg.get("temperature", 0.0)),
            "top_p": float(deterministic_cfg.get("top_p", 1.0)),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        log_lines = [f"[BACKEND] type=openai_compatible url={url} model={self.model_id}"]
        t0 = time.time()
        output_text = ""
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            j = json.loads(body)
            if isinstance(j, dict) and "choices" in j and j["choices"]:
                output_text = str(j["choices"][0].get("text", ""))
            log_lines.append("[BACKEND_OK]")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            log_lines.append(f"[BACKEND_ERROR] http_status={e.code} body={body[:500]}")
        except Exception as e:
            log_lines.append(f"[BACKEND_ERROR] err={type(e).__name__} msg={e}")
        t1 = time.time()
        total_ms = (t1 - t0) * 1000.0
        log_lines.append(f"[BACKEND_METRICS] latency_ms_total={total_ms:.2f}")

        Path(artifacts["output"]).write_text(output_text, encoding="utf-8")
        Path(artifacts["log"]).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return {
            "exit_code": 0,
            "total_ms": total_ms,
            "engine_metrics": {
                "perf": {
                    "prefill_ms": 0.0,
                    "decode_ms": 0.0,
                    "total_ms": total_ms,
                    "tokens_out": len(output_text.split()),
                    "avg_tps": 0.0,
                },
                "amf": {"supported": False, "decision": "unavailable"},
                "mf": {"supported": False},
                "engine": {"mode": "baseline", "accel_enabled": False},
                "spec": {"enabled": False, "reason": "backend_no_spec"},
            },
            "engine_events_path": None,
        }

    def run_draft(
        self,
        prompt: str,
        deterministic_cfg: Dict,
        max_tokens: int,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        _ = (prompt, deterministic_cfg, max_tokens, spec_cfg)
        return {"draft_tokens": [], "state": {}}

    def verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[int],
        deterministic_cfg: Dict,
        spec_cfg: Optional[Dict] = None,
    ) -> Dict:
        _ = (prompt, deterministic_cfg, spec_cfg)
        return {"accepted_count": 0, "verified_logits": None, "rejected_tokens": list(draft_tokens)}

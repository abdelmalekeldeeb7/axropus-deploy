from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


class _EngineParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_int32),
        ("n_batch", ctypes.c_int32),
        ("n_gpu", ctypes.c_int32),
        ("model_path", ctypes.c_char_p),
    ]


class _EngineMetrics(ctypes.Structure):
    _fields_ = [
        ("prefill_ms", ctypes.c_double),
        ("decode_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("restore_ms", ctypes.c_double),
        ("verify_ms", ctypes.c_double),
        ("draft_ms", ctypes.c_double),
        ("tokens_out", ctypes.c_int32),
        ("accepted_tokens", ctypes.c_int32),
        ("rejected_tokens", ctypes.c_int32),
    ]


@dataclass
class EngineClient:
    """
    Optional shared-library boundary for Phase 6.
    If the CUDA engine library is unavailable, caller should use baseline path.
    """

    path: Optional[Path]
    lib: Optional[ctypes.CDLL]

    @property
    def available(self) -> bool:
        return self.lib is not None

    def supports_symbol(self, symbol: str) -> bool:
        if self.lib is None:
            return False
        return hasattr(self.lib, symbol)

    def cuda_probe(self) -> bool:
        if self.lib is None or not hasattr(self.lib, "korith_engine_cuda_probe"):
            return False
        fn = getattr(self.lib, "korith_engine_cuda_probe")
        fn.restype = ctypes.c_bool
        try:
            return bool(fn())
        except Exception:
            return False

    def init_model(self, *, model_path: str, n_ctx: int, n_batch: int, n_gpu: int = 1) -> bool:
        if self.lib is None or not hasattr(self.lib, "korith_engine_init_model"):
            return False
        fn = getattr(self.lib, "korith_engine_init_model")
        fn.argtypes = [ctypes.POINTER(_EngineParams)]
        fn.restype = ctypes.c_bool
        params = _EngineParams(
            n_ctx=int(n_ctx),
            n_batch=int(n_batch),
            n_gpu=int(n_gpu),
            model_path=str(model_path).encode("utf-8"),
        )
        try:
            return bool(fn(ctypes.byref(params)))
        except Exception:
            return False

    def apply_kv_replay(self, blob: bytes) -> bool:
        if self.lib is None or not hasattr(self.lib, "korith_engine_apply_kv_replay"):
            return False
        fn = getattr(self.lib, "korith_engine_apply_kv_replay")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_bool
        if not blob:
            return False
        buf = ctypes.create_string_buffer(blob, len(blob))
        try:
            return bool(fn(ctypes.cast(buf, ctypes.c_void_p), ctypes.c_size_t(len(blob))))
        except Exception:
            return False

    def verify_tokens(self, proposed_tokens: list[int]) -> Optional[int]:
        if self.lib is None or not hasattr(self.lib, "korith_engine_verify_tokens"):
            return None
        fn = getattr(self.lib, "korith_engine_verify_tokens")
        fn.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        fn.restype = ctypes.c_bool
        n_tokens = int(len(proposed_tokens))
        if n_tokens < 0:
            return None
        arr = (ctypes.c_int32 * max(1, n_tokens))(*([int(x) for x in proposed_tokens] or [0]))
        accepted = ctypes.c_int32(0)
        try:
            ok = bool(fn(arr, ctypes.c_int32(n_tokens), ctypes.byref(accepted)))
            return int(accepted.value) if ok else None
        except Exception:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        if self.lib is None or not hasattr(self.lib, "korith_engine_get_metrics"):
            return {}
        fn = getattr(self.lib, "korith_engine_get_metrics")
        fn.argtypes = [ctypes.POINTER(_EngineMetrics)]
        fn.restype = None
        out = _EngineMetrics()
        try:
            fn(ctypes.byref(out))
            return {
                "prefill_ms": float(out.prefill_ms),
                "decode_ms": float(out.decode_ms),
                "total_ms": float(out.total_ms),
                "restore_ms": float(out.restore_ms),
                "verify_ms": float(out.verify_ms),
                "draft_ms": float(out.draft_ms),
                "tokens_out": int(out.tokens_out),
                "accepted_tokens": int(out.accepted_tokens),
                "rejected_tokens": int(out.rejected_tokens),
            }
        except Exception:
            return {}

    def shutdown(self) -> None:
        if self.lib is None or not hasattr(self.lib, "korith_engine_shutdown"):
            return
        fn = getattr(self.lib, "korith_engine_shutdown")
        fn.argtypes = []
        fn.restype = None
        try:
            fn()
        except Exception:
            return


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("KORITH_ENGINE_LIB_PATH", "").strip()
    if env:
        paths.append(Path(env))
    paths.append(Path("./build/libkorith_engine.so"))
    paths.append(Path("./build/lib/libkorith_engine.so"))
    paths.append(Path("./libkorith_engine.so"))
    return paths


def get_engine_client() -> EngineClient:
    for cand in _candidate_paths():
        try:
            if not cand.exists():
                continue
            lib = ctypes.CDLL(str(cand.resolve()))
            return EngineClient(path=cand.resolve(), lib=lib)
        except Exception:
            continue
    return EngineClient(path=None, lib=None)

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional


def _candidate_paths() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("KORITH_ENGINE_LIB_PATH", "").strip()
    if env:
        out.append(Path(env))
    out.append(Path("./build/engine-cuda/libkorith_engine.so"))
    out.append(Path("./build/engine/libkorith_engine.so"))
    out.append(Path("./build/libkorith_engine.so"))
    return out


class CudaKernelBindings:
    def __init__(self, lib: ctypes.CDLL, path: Path) -> None:
        self.lib = lib
        self.path = path

    @staticmethod
    def load() -> Optional["CudaKernelBindings"]:
        for cand in _candidate_paths():
            try:
                if not cand.exists():
                    continue
                lib = ctypes.CDLL(str(cand.resolve()))
                if hasattr(lib, "korith_engine_cuda_probe"):
                    return CudaKernelBindings(lib=lib, path=cand.resolve())
            except Exception:
                continue
        return None

    def cuda_probe(self) -> bool:
        fn = getattr(self.lib, "korith_engine_cuda_probe", None)
        if fn is None:
            return False
        fn.restype = ctypes.c_bool
        try:
            return bool(fn())
        except Exception:
            return False


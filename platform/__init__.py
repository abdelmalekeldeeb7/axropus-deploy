"""Korith platform package.

Note: this package name shadows the stdlib `platform` module when running from
the repo root. To avoid breaking stdlib imports (e.g., uuid -> platform),
we proxy missing attributes to the stdlib module.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path
from types import ModuleType

_STDLIB_PLATFORM: ModuleType | None = None


def _load_stdlib_platform() -> ModuleType:
    global _STDLIB_PLATFORM
    if _STDLIB_PLATFORM is not None:
        return _STDLIB_PLATFORM
    stdlib_path = Path(sysconfig.get_paths()["stdlib"]) / "platform.py"
    spec = importlib.util.spec_from_file_location("_stdlib_platform", stdlib_path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to locate stdlib platform module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STDLIB_PLATFORM = module
    return module


def __getattr__(name: str):
    module = _load_stdlib_platform()
    return getattr(module, name)


def __dir__():
    module = _load_stdlib_platform()
    return sorted(set(globals().keys()) | set(dir(module)))

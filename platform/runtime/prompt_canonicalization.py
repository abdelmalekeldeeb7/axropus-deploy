from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

_TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def canonicalize_prompt_text(prompt: str) -> str:
    text = str(prompt or "")
    # Keep byte-stable prompt hashing across mixed platforms/editors.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    if _env_bool("KORITH_AMF_PROMPT_COLLAPSE_BLANKS", False):
        text = re.sub(r"\n{3,}", "\n\n", text)
    if _env_bool("KORITH_AMF_PROMPT_STRIP_OUTER", False):
        text = text.strip()
    return text


def canonicalize_template_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    normalize_strings = _env_bool("KORITH_AMF_CANONICALIZE_TEMPLATE_STRINGS", False)
    canonicalize_json = _env_bool("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON", False)
    recursive_json = _env_bool("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON_RECURSIVE", False)

    def _normalize_json_value(value: Any) -> Any:
        if isinstance(value, str):
            return canonicalize_prompt_text(value) if normalize_strings else value
        if isinstance(value, dict):
            return {str(k): _normalize_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_json_value(v) for v in value]
        return value

    for k, v in data.items():
        if isinstance(v, str):
            out[str(k)] = canonicalize_prompt_text(v) if normalize_strings else v
            continue
        if canonicalize_json and isinstance(v, (dict, list)):
            payload = _normalize_json_value(v) if recursive_json else v
            out[str(k)] = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            continue
        out[str(k)] = v
    return out

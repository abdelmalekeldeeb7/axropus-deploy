from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.runtime.prompt_canonicalization import canonicalize_prompt_text, canonicalize_template_data


class PromptCanonicalizationTests(unittest.TestCase):
    def test_canonicalize_prompt_text_normalizes_line_endings_and_trailing_spaces(self):
        raw = "A  \r\nB\t \r\nC\r\n"
        out = canonicalize_prompt_text(raw)
        self.assertEqual(out, "A\nB\nC\n")

    def test_canonicalize_template_data_json_is_opt_in(self):
        old = os.environ.get("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON")
        try:
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = "1"
            out = canonicalize_template_data({"x": {"b": 1, "a": 2}})
            self.assertEqual(out["x"], "{\"a\":2,\"b\":1}")
        finally:
            if old is None:
                os.environ.pop("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON", None)
            else:
                os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = old

    def test_canonicalize_template_data_preserves_json_when_disabled(self):
        old = os.environ.get("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON")
        try:
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = "0"
            value = {"b": 1, "a": 2}
            out = canonicalize_template_data({"x": value})
            self.assertIs(out["x"], value)
        finally:
            if old is None:
                os.environ.pop("KORITH_AMF_CANONICALIZE_TEMPLATE_JSON", None)
            else:
                os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = old

    def test_canonicalize_template_data_recursive_json_normalization_is_opt_in(self):
        keys = (
            "KORITH_AMF_CANONICALIZE_TEMPLATE_JSON",
            "KORITH_AMF_CANONICALIZE_TEMPLATE_JSON_RECURSIVE",
            "KORITH_AMF_CANONICALIZE_TEMPLATE_STRINGS",
        )
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON"] = "1"
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_JSON_RECURSIVE"] = "1"
            os.environ["KORITH_AMF_CANONICALIZE_TEMPLATE_STRINGS"] = "1"
            out = canonicalize_template_data(
                {
                    "x": {
                        "msg": "A  \r\nB\t \r\n",
                        "nested": [{"k": " C \r\n"}],
                    }
                }
            )
            self.assertEqual(out["x"], "{\"msg\":\"A\\nB\\n\",\"nested\":[{\"k\":\" C\\n\"}]}")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()

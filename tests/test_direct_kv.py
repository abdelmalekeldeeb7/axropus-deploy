"""test_direct_kv.py — Integration tests for the direct-GPU KV save/restore path.

Tests:
  1. Cold run with KORITH_AMF_DIRECT_GPU=1 creates a .kv file with AMFK magic.
  2. Warm run gets an AMF hit and restore_ms < 5000 ms.
  3. Output tokens on warm run match cold run (correctness).
  4. Legacy (.kv with llama.cpp magic) blobs are still restored correctly when
     KORITH_AMF_DIRECT_GPU=0.

Usage:
  pytest tests/test_direct_kv.py -v
  # or point at a real binary:
  KORITH_BIN=/path/to/korith_dynamic pytest tests/test_direct_kv.py -v

Environment:
  KORITH_BIN        — path to korith_dynamic binary (default: ./build/korith_dynamic)
  KORITH_MODEL      — path to .gguf model file (required for live tests)
  KORITH_AMF_PATH   — override AMF store path (default: /tmp/test_amf_store)
  KORITH_TEST_LIVE  — set to 1 to run live inference tests (needs GPU + model)
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

# ── Constants ─────────────────────────────────────────────────────────────────

AMF_DIRECT_MAGIC = 0x414D464B  # "AMFK"
AMF_DIRECT_VERSION = 1
AMF_HEADER_SIZE = 56  # sizeof(AmfDirectKvHeader)

KORITH_BIN = os.environ.get(
    "KORITH_BIN",
    str(Path(__file__).parent.parent / "build" / "korith_dynamic"),
)
KORITH_MODEL = os.environ.get("KORITH_MODEL", "")
RUN_LIVE = os.environ.get("KORITH_TEST_LIVE", "0").strip() == "1"

# Short prompt that is long enough to exceed AMF min_tokens (default 64).
TEST_PROMPT = (
    "Evaluate the implications of transformer attention complexity scaling "
    "with sequence length and describe three architectural approaches that "
    "reduce this complexity while preserving model quality. "
) * 4  # ~300 tokens; keeps test fast


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_amf_direct_header(data: bytes) -> dict:
    """Parse AmfDirectKvHeader from raw bytes."""
    assert len(data) >= AMF_HEADER_SIZE, f"blob too small: {len(data)} bytes"
    magic, version, n_layers, n_tokens, n_kv_heads, head_dim, dtype, reserved, \
        total_kv_bytes, model_hash, prefix_hash = struct.unpack_from(
            "<IIIIIIII QQQ", data, 0
        )
    return {
        "magic": magic,
        "version": version,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype,
        "total_kv_bytes": total_kv_bytes,
        "model_hash": model_hash,
        "prefix_hash": prefix_hash,
    }


def find_kv_files(amf_path: str) -> list[Path]:
    return sorted(Path(amf_path).glob("*.kv"))


def run_korith(
    prompt: str,
    amf_path: str,
    *,
    direct_gpu: bool = False,
    max_tokens: int = 32,
    extra_env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["KORITH_ENABLE_AMF"] = "1"
    env["KORITH_AMF_PATH"] = amf_path
    env["KORITH_AMF_DIRECT_GPU"] = "1" if direct_gpu else "0"
    env["KORITH_AMF_MIN_TOKENS"] = "64"
    if extra_env:
        env.update(extra_env)

    cmd = [
        KORITH_BIN,
        "--model", KORITH_MODEL,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--no-color",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    return result


def extract_log_field(stderr: str, tag: str, field: str) -> Optional[str]:
    """Extract a field value from a structured log line.
    E.g. extract_log_field(stderr, "AMF_SKIP", "restore_ms") → "1234.56"
    """
    pattern = rf"\[{re.escape(tag)}\].*\b{re.escape(field)}=([^\s]+)"
    m = re.search(pattern, stderr, re.MULTILINE)
    return m.group(1) if m else None


# ── Unit tests (no GPU / binary required) ────────────────────────────────────

class TestHeaderParsing:
    """Parse AMFK header bytes without running any binary."""

    def _make_header(self, **kwargs) -> bytes:
        defaults = dict(
            magic=AMF_DIRECT_MAGIC,
            version=AMF_DIRECT_VERSION,
            n_layers=32,
            n_tokens=1024,
            n_kv_heads=8,
            head_dim=128,
            dtype=0,   # f16
            reserved=0,
            total_kv_bytes=1024 * 1024 * 1024,
            model_hash=0xDEADBEEFCAFEBABE,
            prefix_hash=0x0102030405060708,
        )
        defaults.update(kwargs)
        return struct.pack(
            "<IIIIIIII QQQ",
            defaults["magic"],
            defaults["version"],
            defaults["n_layers"],
            defaults["n_tokens"],
            defaults["n_kv_heads"],
            defaults["head_dim"],
            defaults["dtype"],
            defaults["reserved"],
            defaults["total_kv_bytes"],
            defaults["model_hash"],
            defaults["prefix_hash"],
        )

    def test_magic_recognized(self):
        blob = self._make_header()
        hdr = parse_amf_direct_header(blob)
        assert hdr["magic"] == AMF_DIRECT_MAGIC

    def test_version_field(self):
        blob = self._make_header(version=1)
        hdr = parse_amf_direct_header(blob)
        assert hdr["version"] == 1

    def test_wrong_magic_not_amfk(self):
        # Legacy llama.cpp KV blobs have a different magic.
        blob = self._make_header(magic=0x31464D41)  # "AMF1" (index magic, not KV)
        hdr = parse_amf_direct_header(blob)
        assert hdr["magic"] != AMF_DIRECT_MAGIC

    def test_header_size_56_bytes(self):
        """Header must be exactly 56 bytes (verified by static_assert in C++)."""
        blob = self._make_header()
        assert len(blob) == AMF_HEADER_SIZE

    def test_roundtrip_fields(self):
        blob = self._make_header(n_layers=80, n_tokens=85_000, n_kv_heads=4, head_dim=256)
        hdr = parse_amf_direct_header(blob)
        assert hdr["n_layers"] == 80
        assert hdr["n_tokens"] == 85_000
        assert hdr["n_kv_heads"] == 4
        assert hdr["head_dim"] == 256


# ── Live inference tests (require GPU + binary + model) ──────────────────────

@pytest.mark.skipif(not RUN_LIVE, reason="Set KORITH_TEST_LIVE=1 to enable live tests")
@pytest.mark.skipif(not KORITH_MODEL, reason="KORITH_MODEL env var not set")
class TestDirectKvLive:
    """End-to-end tests using the real korith_dynamic binary with CUDA KV path."""

    @pytest.fixture(autouse=True)
    def amf_dir(self, tmp_path):
        """Provide a fresh AMF store directory per test."""
        self._amf_path = str(tmp_path / "amf_store")
        os.makedirs(self._amf_path, exist_ok=True)
        yield self._amf_path
        shutil.rmtree(self._amf_path, ignore_errors=True)

    def test_cold_run_creates_amfk_file(self, amf_dir):
        """Cold run with KORITH_AMF_DIRECT_GPU=1 must produce a .kv with AMFK magic."""
        result = run_korith(TEST_PROMPT, amf_dir, direct_gpu=True, max_tokens=16)
        assert result.returncode == 0, f"korith failed:\n{result.stderr}"

        kv_files = find_kv_files(amf_dir)
        assert len(kv_files) >= 1, "expected at least one .kv file in AMF store"

        # Every .kv written by direct-GPU path must start with AMFK magic.
        for kv_file in kv_files:
            data = kv_file.read_bytes()
            assert len(data) >= AMF_HEADER_SIZE, f"{kv_file} too small"
            hdr = parse_amf_direct_header(data)
            assert hdr["magic"] == AMF_DIRECT_MAGIC, (
                f"{kv_file} has wrong magic 0x{hdr['magic']:08X} (expected AMFK)"
            )
            assert hdr["version"] == AMF_DIRECT_VERSION
            assert hdr["n_layers"] > 0
            assert hdr["n_tokens"] > 0

    def test_warm_run_hits_and_restores_fast(self, amf_dir):
        """Second run must get an AMF hit and restore in < 5000 ms."""
        # Cold run — populate the store.
        cold = run_korith(TEST_PROMPT, amf_dir, direct_gpu=True, max_tokens=16)
        assert cold.returncode == 0, f"cold run failed:\n{cold.stderr}"
        assert "[AMF_STATS]" in cold.stderr, "no AMF_STATS in cold run"

        # Warm run — should hit the cache.
        warm = run_korith(TEST_PROMPT, amf_dir, direct_gpu=True, max_tokens=16)
        assert warm.returncode == 0, f"warm run failed:\n{warm.stderr}"

        assert "[AMF_HIT]" in warm.stderr, (
            f"Expected AMF_HIT in warm run stderr.\n{warm.stderr}"
        )
        assert "[AMF_SKIP]" in warm.stderr, "Expected AMF_SKIP in warm run"

        restore_ms_str = extract_log_field(warm.stderr, "AMF_SKIP", "restore_ms")
        assert restore_ms_str is not None, "restore_ms not found in AMF_SKIP line"

        restore_ms = float(restore_ms_str)
        assert restore_ms < 5000.0, (
            f"restore_ms={restore_ms:.1f} exceeds 5000 ms target for direct-GPU path"
        )

    def test_output_tokens_match_legacy_path(self, amf_dir):
        """Direct-GPU restore must produce the same tokens as the legacy path."""
        # Run with legacy path (no direct GPU) to populate and get baseline output.
        legacy_dir = str(Path(amf_dir) / "legacy")
        os.makedirs(legacy_dir, exist_ok=True)
        direct_dir = str(Path(amf_dir) / "direct")
        os.makedirs(direct_dir, exist_ok=True)

        # Cold run — legacy.
        leg_cold = run_korith(TEST_PROMPT, legacy_dir, direct_gpu=False, max_tokens=32)
        assert leg_cold.returncode == 0
        legacy_cold_out = leg_cold.stdout.strip()

        # Cold run — direct GPU (populates direct store).
        dir_cold = run_korith(TEST_PROMPT, direct_dir, direct_gpu=True, max_tokens=32)
        assert dir_cold.returncode == 0

        # Warm run — direct GPU (should restore and generate).
        dir_warm = run_korith(TEST_PROMPT, direct_dir, direct_gpu=True, max_tokens=32)
        assert dir_warm.returncode == 0
        direct_warm_out = dir_warm.stdout.strip()

        # Outputs should match modulo leading/trailing whitespace.
        assert legacy_cold_out == direct_warm_out, (
            f"Output mismatch between legacy cold and direct warm.\n"
            f"Legacy: {legacy_cold_out!r}\n"
            f"Direct: {direct_warm_out!r}"
        )

    def test_legacy_blobs_restored_without_direct_gpu(self, amf_dir):
        """Legacy llama.cpp blobs must still be restored when KORITH_AMF_DIRECT_GPU=0."""
        # Cold run — legacy (no direct GPU).
        cold = run_korith(TEST_PROMPT, amf_dir, direct_gpu=False, max_tokens=16)
        assert cold.returncode == 0

        kv_files = find_kv_files(amf_dir)
        assert len(kv_files) >= 1

        # Confirm it is NOT an AMFK blob.
        for kv_file in kv_files:
            data = kv_file.read_bytes()
            if len(data) >= 4:
                magic = struct.unpack_from("<I", data, 0)[0]
                assert magic != AMF_DIRECT_MAGIC, (
                    f"Legacy path wrote AMFK magic in {kv_file}"
                )

        # Warm run — legacy path should restore from the legacy blob.
        warm = run_korith(TEST_PROMPT, amf_dir, direct_gpu=False, max_tokens=16)
        assert warm.returncode == 0
        assert "[AMF_HIT]" in warm.stderr, (
            f"Expected AMF_HIT on warm legacy run.\n{warm.stderr}"
        )

    def test_direct_gpu_env_flag_logged(self, amf_dir):
        """KORITH_AMF_DIRECT_GPU=1 must emit [AMF_DIRECT_GPU] enabled log line."""
        result = run_korith(TEST_PROMPT, amf_dir, direct_gpu=True, max_tokens=4)
        assert result.returncode == 0
        assert "[AMF_DIRECT_GPU] enabled" in result.stderr, (
            f"Expected '[AMF_DIRECT_GPU] enabled' in stderr.\n{result.stderr}"
        )

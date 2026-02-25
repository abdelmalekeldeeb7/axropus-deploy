#!/usr/bin/env python3
from __future__ import annotations

"""
Live benchmark harness.

What this script guarantees (audit-proof):
- Counts only tokens that were actually emitted on stdout by the worker process.
- Uses a monotonic, high-resolution clock (perf_counter_ns) for TPS timing.
- Computes TPS from observed output tokens (no estimates, no synthetic metrics).

How it works:
- Parent process spawns a worker subprocess.
- Worker loads the engine shared library and streams output back to the parent using a
  simple stdout framing protocol:
    - Data frame:  b"D <len>\\n" + <len raw bytes>
    - Token frame: b"T <tokens_total>\\n"
    - End frame:   b"E\\n"
- Parent parses worker stdout, forwards raw output bytes to its own stdout, and
  computes sustained TPS from token frames.

Backends:
- "capi": uses legacy symbols exported by `korith_c_api` (korith_init/tokenize/step).
          This backend prints *token IDs* (one per line) as the observable stdout stream,
          enabling exact token counting without detokenization.
- "core": uses the newer `engine_init/engine_step/engine_shutdown` ABI if present.
          This backend forwards the engine's real text output; token counts are derived
          from `engine_step()`'s return value (printed tokens).

Note: The current `core` engine in this repo (as of v0) does not ingest prompts; prompt
feeding is fully supported in the "capi" backend.
"""

import argparse
from collections import deque
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import BinaryIO, Deque, Iterable, Optional, Tuple


def _read_exact(f: BinaryIO, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            raise EOFError(f"unexpected EOF (wanted {n} bytes, got {len(out)})")
        out += chunk
    return bytes(out)


def _write_all_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _frame_data(fd: int, payload: bytes) -> None:
    _write_all_fd(fd, f"D {len(payload)}\n".encode("ascii"))
    if payload:
        _write_all_fd(fd, payload)


def _frame_tokens(fd: int, tokens_total: int) -> None:
    _write_all_fd(fd, f"T {int(tokens_total)}\n".encode("ascii"))


def _frame_end(fd: int) -> None:
    _write_all_fd(fd, b"E\n")


class RollingTps:
    def __init__(self, window_s: float) -> None:
        self._window_ns = int(max(1e-6, float(window_s)) * 1e9)
        self._samples: Deque[Tuple[int, int]] = deque()

    def add(self, t_ns: int, tokens_total: int) -> None:
        if self._samples and tokens_total < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((int(t_ns), int(tokens_total)))

        cutoff = t_ns - self._window_ns
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def rolling(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        t0, tok0 = self._samples[0]
        t1, tok1 = self._samples[-1]
        dt = (t1 - t0) * 1e-9
        if dt <= 1e-12:
            return 0.0
        d = tok1 - tok0
        return 0.0 if d <= 0 else (d / dt)

    def instant(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        t0, tok0 = self._samples[-2]
        t1, tok1 = self._samples[-1]
        dt = (t1 - t0) * 1e-9
        if dt <= 1e-12:
            return 0.0
        d = tok1 - tok0
        return 0.0 if d <= 0 else (d / dt)


def _lib_has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


def _load_lib(path: Path) -> ctypes.CDLL:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return ctypes.CDLL(str(path))


def _worker_run_capi(
    lib: ctypes.CDLL,
    comm_fd: int,
    model: str,
    n_ctx: int,
    max_tokens_total: int,
    max_tokens_per_prompt: int,
) -> int:
    lib.korith_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.korith_init.restype = ctypes.c_bool
    lib.korith_tokenize.argtypes = [ctypes.c_char_p]
    lib.korith_tokenize.restype = ctypes.c_int
    lib.korith_step.argtypes = []
    lib.korith_step.restype = ctypes.c_int
    lib.korith_shutdown.argtypes = []
    lib.korith_shutdown.restype = None

    model_b = model.encode("utf-8")
    if b"\x00" in model_b:
        raise ValueError("model path contains NUL byte")

    ok = bool(lib.korith_init(model_b, int(n_ctx)))
    if not ok:
        return 1

    tokens_total = 0
    try:
        flush_every = 256  # write in batches to reduce overhead, still exact.

        while tokens_total < max_tokens_total:
            line = sys.stdin.buffer.readline()
            if not line:
                break
            prompt = line.rstrip(b"\r\n")
            if b"\x00" in prompt:
                break

            n_prompt = int(lib.korith_tokenize(prompt))
            if n_prompt < 0:
                break

            pending_lines: list[bytes] = []
            produced = 0
            while produced < max_tokens_per_prompt and tokens_total < max_tokens_total:
                tok = int(lib.korith_step())
                if tok < 0:
                    break
                produced += 1
                tokens_total += 1
                pending_lines.append(str(tok).encode("ascii"))

                if len(pending_lines) >= flush_every:
                    payload = b"\n".join(pending_lines) + b"\n"
                    pending_lines.clear()
                    _frame_data(comm_fd, payload)
                    _frame_tokens(comm_fd, tokens_total)

            if pending_lines:
                payload = b"\n".join(pending_lines) + b"\n"
                pending_lines.clear()
                _frame_data(comm_fd, payload)
                _frame_tokens(comm_fd, tokens_total)

            # Prompt boundary marker (non-token).
            _frame_data(comm_fd, b"#EOP\n")
    finally:
        try:
            lib.korith_shutdown()
        except Exception:
            pass

    _frame_tokens(comm_fd, tokens_total)
    _frame_end(comm_fd)
    return 0


def _worker_run_core(
    lib: ctypes.CDLL,
    comm_fd: int,
    model: str,
    batch_tokens: int,
    max_tokens_total: int,
) -> int:
    lib.engine_init.argtypes = [ctypes.c_char_p]
    lib.engine_init.restype = ctypes.c_bool
    lib.engine_step.argtypes = [ctypes.c_int]
    lib.engine_step.restype = ctypes.c_int32
    lib.engine_shutdown.argtypes = []
    lib.engine_shutdown.restype = None

    model_b = model.encode("utf-8")
    if b"\x00" in model_b:
        raise ValueError("model path contains NUL byte")

    # Redirect worker stdout (fd=1) into a pipe so we can forward it as data frames.
    # Keep a duplicate of the original stdout for framed communication to the parent.
    rfd, wfd = os.pipe()
    os.dup2(wfd, 1)
    os.close(wfd)

    stop_read = False

    def pump_stdout() -> None:
        nonlocal stop_read
        try:
            while True:
                chunk = os.read(rfd, 1 << 16)
                if not chunk:
                    break
                _frame_data(comm_fd, chunk)
        except BrokenPipeError:
            stop_read = True
        finally:
            try:
                os.close(rfd)
            except Exception:
                pass

    t = __import__("threading").Thread(target=pump_stdout, name="korith-stdout-pump", daemon=True)
    t.start()

    ok = bool(lib.engine_init(model_b))
    if not ok:
        # Restore stdout to avoid leaving fd=1 broken for teardown.
        os.dup2(comm_fd, 1)
        t.join()
        _frame_end(comm_fd)
        return 1

    tokens_total = 0
    try:
        while tokens_total < max_tokens_total and not stop_read:
            printed = int(lib.engine_step(int(batch_tokens)))
            if printed < 0:
                break
            if printed == 0:
                break
            tokens_total += printed
            _frame_tokens(comm_fd, tokens_total)
    finally:
        try:
            lib.engine_shutdown()
        except Exception:
            pass

    # Close the pipe write end by restoring stdout (dup2 closes the previous fd=1).
    os.dup2(comm_fd, 1)
    t.join()
    _frame_tokens(comm_fd, tokens_total)
    _frame_end(comm_fd)
    return 0


def _worker_main(args: argparse.Namespace) -> int:
    lib_path = Path(args.lib).resolve()
    lib = _load_lib(lib_path)

    has_core = all(_lib_has_symbol(lib, s) for s in ("engine_init", "engine_step", "engine_shutdown"))
    has_capi = all(_lib_has_symbol(lib, s) for s in ("korith_init", "korith_tokenize", "korith_step", "korith_shutdown"))

    backend = args.backend
    if backend == "auto":
        backend = "core" if has_core else "capi"

    if backend == "core" and not has_core:
        raise RuntimeError("backend=core requested but engine_* symbols were not found in the shared library")
    if backend == "capi" and not has_capi:
        raise RuntimeError("backend=capi requested but korith_* symbols were not found in the shared library")

    comm_fd = os.dup(1)
    try:
        if backend == "core":
            return _worker_run_core(
                lib,
                comm_fd,
                model=args.model,
                batch_tokens=args.batch_tokens,
                max_tokens_total=args.max_tokens_total,
            )
        return _worker_run_capi(
            lib,
            comm_fd,
            model=args.model,
            n_ctx=args.n_ctx,
            max_tokens_total=args.max_tokens_total,
            max_tokens_per_prompt=args.max_tokens_per_prompt,
        )
    finally:
        try:
            os.close(comm_fd)
        except Exception:
            pass


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        p = Path(args.prompts_file)
        prompts.extend([line.rstrip("\r\n") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()])
    if not prompts:
        prompts = ["Hello from Korith"]
        print("note: no prompts provided; using a default prompt", file=sys.stderr)
    return prompts


def _parent_main(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--lib",
        str(args.lib),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--n-ctx",
        str(args.n_ctx),
        "--batch-tokens",
        str(args.batch_tokens),
        "--max-tokens-total",
        str(args.max_tokens_total),
        "--max-tokens-per-prompt",
        str(args.max_tokens_per_prompt),
    ]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
    )

    assert p.stdin is not None
    assert p.stdout is not None

    prompts = list(_iter_prompts(args))
    for _ in range(int(args.repeat_count)):
        for pr in prompts:
            p.stdin.write(pr.encode("utf-8") + b"\n")
    p.stdin.close()

    tps = RollingTps(args.window_s)
    report_every_ns = int(max(1e-6, float(args.report_every_s)) * 1e9)
    last_report_ns = time.perf_counter_ns()
    start_ns = last_report_ns
    last_tokens = 0

    try:
        while True:
            header = p.stdout.readline()
            if not header:
                break

            if header.startswith(b"D "):
                n = int(header[2:].strip() or b"0")
                payload = _read_exact(p.stdout, n)
                if args.echo_output:
                    sys.stdout.buffer.write(payload)
                continue

            if header.startswith(b"T "):
                tokens_total = int(header[2:].strip() or b"0")
                now_ns = time.perf_counter_ns()
                tps.add(now_ns, tokens_total)

                if now_ns - last_report_ns >= report_every_ns:
                    inst = tps.instant()
                    roll = tps.rolling()
                    dt_s = (now_ns - start_ns) * 1e-9
                    print(
                        f"tokens={tokens_total} (+{tokens_total - last_tokens}) "
                        f"t={dt_s:.2f}s inst_tps={inst:.1f} roll_tps={roll:.1f}",
                        file=sys.stderr,
                    )
                    last_report_ns = now_ns
                    last_tokens = tokens_total
                continue

            if header == b"E\n":
                break
    except KeyboardInterrupt:
        try:
            p.terminate()
        except Exception:
            pass
    finally:
        try:
            rc = p.wait()
        except Exception:
            rc = -1

    end_ns = time.perf_counter_ns()
    total_s = (end_ns - start_ns) * 1e-9
    roll = tps.rolling()
    if args.echo_output:
        try:
            sys.stdout.buffer.flush()
        except Exception:
            pass
    print(f"done: rc={rc} seconds={total_s:.3f} roll_tps={roll:.2f}", file=sys.stderr)
    return 0 if rc == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Korith live TPS benchmark (stdout-parsed, audit-proof)")
    ap.add_argument("--lib", default=str(Path(__file__).resolve().parents[2] / "build" / "libkorith.so"))
    ap.add_argument("--backend", choices=("auto", "capi", "core"), default="auto")
    ap.add_argument("--model", required=True, help="Path to a GGUF model")
    ap.add_argument("--n-ctx", type=int, default=4096, help="(capi) context size")
    ap.add_argument("--batch-tokens", type=int, default=256, help="(core) tokens per engine_step() call")
    ap.add_argument("--max-tokens-total", type=int, default=4096, help="Stop after this many output tokens")
    ap.add_argument("--max-tokens-per-prompt", type=int, default=256, help="(capi) generation limit per prompt")
    ap.add_argument("--prompt", action="append", help="Prompt to send (repeatable)")
    ap.add_argument("--prompts-file", help="File containing prompts (1 per line)")
    ap.add_argument("--repeat-count", type=int, default=1, help="Repeat the prompt list N times")
    ap.add_argument("--window-s", type=float, default=5.0, help="Rolling TPS window in seconds")
    ap.add_argument("--report-every-s", type=float, default=0.25, help="Status print period (stderr)")
    ap.add_argument(
        "--no-echo-output",
        action="store_true",
        help="Do not forward worker output bytes to stdout (metrics only)",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.echo_output = not bool(args.no_echo_output)

    if args.worker:
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

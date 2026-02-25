from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

from power import PowerMonitor
from physics_control import PhysicsConfig, PhysicsController


@dataclass
class RuntimeMetrics:
    """
    In-memory metrics snapshot (updated once per second).
    """

    t_s: float = 0.0
    tokens_total: int = 0
    tps: float = 0.0
    watts: float = 0.0
    tok_per_j: float = 0.0
    temp_c: float = 0.0
    N: int = 1
    Q: float = 0.0
    J: float = 0.0
    tau_setpoint: float = 0.0


class SharedMetricsBuffer:
    """
    Thread-safe shared metrics state between the inference loop and policy loop.

    - Inference loop: increments `tokens_total`.
    - Policy loop: publishes full `RuntimeMetrics` snapshots.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens_total = 0
        self._latest: RuntimeMetrics | None = None

    def add_tokens(self, n: int = 1) -> int:
        with self._lock:
            self._tokens_total += int(n)
            return self._tokens_total

    def tokens_total(self) -> int:
        with self._lock:
            return self._tokens_total

    def publish(self, metrics: RuntimeMetrics) -> None:
        with self._lock:
            self._latest = metrics

    def latest(self) -> RuntimeMetrics | None:
        with self._lock:
            return self._latest


LATEST_METRICS: RuntimeMetrics | None = None


class RateLimiter:
    """
    Token rate limiter (tokens/sec).

    A target of 0 means "unthrottled". The limiter is enforced in the controller
    thread and updated by the metrics/policy thread.
    """

    def __init__(self, *, target_tps: float = 0.0) -> None:
        self._target_tps = float(target_tps)
        self._next_s = time.perf_counter()
        self._cv = threading.Condition()

    def set_target_tps(self, target_tps: float) -> None:
        with self._cv:
            self._target_tps = max(0.0, float(target_tps))
            now = time.perf_counter()
            if self._next_s < now:
                self._next_s = now
            self._cv.notify_all()

    def notify(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def wait(self, stop: threading.Event) -> bool:
        with self._cv:
            while True:
                if stop.is_set():
                    return False

                tps = self._target_tps
                if tps <= 0.0:
                    return True

                interval = 1.0 / tps
                now = time.perf_counter()
                if now >= self._next_s:
                    self._next_s = now + interval
                    return True

                timeout = self._next_s - now
                self._cv.wait(timeout=min(timeout, 0.25))


def load_korith_lib() -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[1]
    lib_path = root / "build" / "libkorith.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"libkorith.so not found at {lib_path} (build the C API first)")

    lib = ctypes.CDLL(str(lib_path))

    # Fail fast if expected symbols are missing.
    required = ("korith_init", "korith_tokenize", "korith_step", "korith_shutdown")
    for name in required:
        if not hasattr(lib, name):
            raise AttributeError(f"Missing symbol in libkorith.so: {name}")

    lib.korith_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.korith_init.restype = ctypes.c_bool

    lib.korith_tokenize.argtypes = [ctypes.c_char_p]
    lib.korith_tokenize.restype = ctypes.c_int

    lib.korith_step.argtypes = []
    lib.korith_step.restype = ctypes.c_int

    lib.korith_shutdown.argtypes = []
    lib.korith_shutdown.restype = None

    return lib


def metrics_reporter(
    stop: threading.Event,
    done: threading.Event,
    shared: SharedMetricsBuffer,
    pm: PowerMonitor,
    phys: PhysicsController,
    limiter: RateLimiter,
    *,
    window_s: float = 5.0,
    policy_period_s: float = 0.2,
    print_period_s: float = 1.0,
) -> None:
    """
    Policy loop: periodically samples metrics, updates the policy, and publishes
    snapshots to a shared buffer.

    This is intentionally decoupled from the inference loop.
    """
    history = deque()  # (t_s, tokens_total, joules_total)
    last_print_s = 0.0
    last_tick_s: float | None = None

    while not stop.is_set() and not done.is_set():
        now_s = time.perf_counter()
        dt_s = policy_period_s if last_tick_s is None else max(0.0, now_s - last_tick_s)
        last_tick_s = now_s
        tokens_total = shared.tokens_total()

        p = pm.snapshot()
        history.append((now_s, tokens_total, p.joules_total))

        cutoff = now_s - window_s
        while history and history[0][0] < cutoff:
            history.popleft()

        tps = 0.0
        tok_per_j = 0.0
        if len(history) >= 2:
            t0, tok0, j0 = history[0]
            t1, tok1, j1 = history[-1]
            dt = t1 - t0
            d_tok = tok1 - tok0
            d_j = j1 - j0
            if dt > 1e-9:
                tps = d_tok / dt
            if d_j > 1e-9:
                tok_per_j = d_tok / d_j

        # 1) SYSTEM STATE (MANDATORY)
        # x = {T, P, tau, N, Q}
        phys_step = phys.step(
            T_meas_c=p.temperature_c,
            P_meas_w=p.watts_avg,
            tau_meas_tps=tps,
            tokens_total=tokens_total,
            dt_s=dt_s,
        )

        limiter.set_target_tps(phys_step.tau_setpoint)

        snapshot = RuntimeMetrics(
            t_s=now_s,
            tokens_total=tokens_total,
            tps=tps,
            watts=p.watts_avg,
            tok_per_j=tok_per_j,
            temp_c=p.temperature_c,
            N=phys_step.x.N,
            Q=phys_step.x.Q,
            J=phys_step.J,
            tau_setpoint=phys_step.tau_setpoint,
        )

        shared.publish(snapshot)
        global LATEST_METRICS
        LATEST_METRICS = snapshot

        if (now_s - last_print_s) >= print_period_s:
            # 11) OUTPUT: Print live metrics
            # TPS | Watts | Tok/J | Temp | N | Q | Objective
            print(
                f"{tps:8.1f} | {p.watts_avg:6.1f} | {tok_per_j:8.4f} | {p.temperature_c:5.1f} |"
                f" {phys_step.x.N:2d} | {phys_step.x.Q:4.2f} | {phys_step.J:8.2f}"
            )
            last_print_s = now_s

        if stop.wait(timeout=policy_period_s):
            break


def inference_worker(
    stop: threading.Event,
    done: threading.Event,
    lib: ctypes.CDLL,
    limiter: RateLimiter,
    shared: SharedMetricsBuffer,
    *,
    max_tokens: int,
) -> None:
    """
    Inference loop: calls korith_step() token-by-token.

    This loop never calls NVML and never runs policy logic.
    """
    try:
        tokens_total = 0
        while not stop.is_set() and tokens_total < max_tokens:
            if not limiter.wait(stop):
                break
            tok = lib.korith_step()
            if tok < 0:
                break
            tokens_total = shared.add_tokens(1)
    finally:
        done.set()


def main() -> int:
    ap = argparse.ArgumentParser(description="Korith Python control plane (ctypes over libkorith.so)")
    ap.add_argument("--model", required=True, help="Path to a GGUF model")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    ap.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    ap.add_argument("--gpu", type=int, default=0, help="NVML device index")
    args = ap.parse_args()

    try:
        lib = load_korith_lib()
    except Exception as e:
        print(f"error: failed to load libkorith.so: {e}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"error: model file not found: {model_path}", file=sys.stderr)
        return 2

    if args.n_ctx <= 0:
        print("error: --n-ctx must be > 0", file=sys.stderr)
        return 2

    pm = PowerMonitor(device_index=args.gpu)

    stop = threading.Event()
    done = threading.Event()
    reporter: threading.Thread | None = None
    infer: threading.Thread | None = None
    limiter = RateLimiter(target_tps=0.0)
    shared = SharedMetricsBuffer()

    t0 = time.perf_counter()

    try:
        model_b = str(model_path).encode("utf-8")
        if b"\x00" in model_b:
            print("error: model path contains NUL byte", file=sys.stderr)
            return 2

        ok = lib.korith_init(model_b, int(args.n_ctx))
        if not ok:
            print("error: korith_init failed", file=sys.stderr)
            return 1

        prompt_b = args.prompt.encode("utf-8")
        if b"\x00" in prompt_b:
            print("error: prompt contains NUL byte", file=sys.stderr)
            return 2

        n_prompt = lib.korith_tokenize(prompt_b)
        if n_prompt < 0:
            print("error: korith_tokenize failed", file=sys.stderr)
            return 1

        pm.start()

        phys_cfg = PhysicsConfig(dt_s=0.1)
        phys = PhysicsController(phys_cfg, ctx_max=int(args.n_ctx), prompt_tokens=int(n_prompt))

        print("TPS | Watts | Tok/J | Temp | N | Q | Objective")

        reporter = threading.Thread(
            target=metrics_reporter,
            args=(stop, done, shared, pm, phys, limiter),
            kwargs={"window_s": 5.0, "policy_period_s": phys_cfg.dt_s, "print_period_s": 1.0},
            name="korith-policy",
            daemon=False,
        )
        reporter.start()

        infer = threading.Thread(
            target=inference_worker,
            args=(stop, done, lib, limiter, shared),
            kwargs={"max_tokens": int(args.max_tokens)},
            name="korith-infer",
            daemon=False,
        )
        infer.start()

        try:
            done.wait()
        except KeyboardInterrupt:
            stop.set()
    finally:
        stop.set()
        limiter.notify()
        if infer is not None:
            infer.join()
        if reporter is not None:
            reporter.join()
        try:
            pm.shutdown()
        except Exception:
            pass
        try:
            lib.korith_shutdown()
        except Exception:
            pass

    dt = time.perf_counter() - t0
    tokens_total = shared.tokens_total()
    tps = tokens_total / max(1e-9, dt)
    print(f"done: tokens={tokens_total} seconds={dt:.3f} tps={tps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

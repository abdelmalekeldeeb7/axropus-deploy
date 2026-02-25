from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Deque, Tuple

try:
    import pynvml
except ModuleNotFoundError:  # pragma: no cover
    pynvml = None


@dataclass
class PowerSnapshot:
    """
    A point-in-time view of GPU power/energy.

    All timestamps are from time.perf_counter() (monotonic).
    """

    t_s: float
    watts: float
    watts_avg: float
    joules_total: float
    temperature_c: float


@dataclass
class PowerMonitor:
    device_index: int = 0
    window_s: float = 2.0
    sample_interval_s: float = 0.1
    _handle: object | None = None
    _samples: Deque[Tuple[float, float]] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _nvml_inited: bool = field(default=False, init=False)

    _joules_total: float = field(default=0.0, init=False)
    _last_t_s: float | None = field(default=None, init=False)
    _last_watts: float | None = field(default=None, init=False)
    _last_temp_c: float | None = field(default=None, init=False)

    def start(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml is not installed (pip install pynvml)")
        if self._thread is not None:
            return

        pynvml.nvmlInit()
        self._nvml_inited = True
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="korith-power", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None

        with self._lock:
            self._handle = None
            self._samples.clear()
            self._joules_total = 0.0
            self._last_t_s = None
            self._last_watts = None
            self._last_temp_c = None

        if pynvml is None:
            return
        if self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._nvml_inited = False

    def _read_instant_watts(self) -> float:
        if self._handle is None:
            raise RuntimeError("PowerMonitor not started")
        mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
        return float(mw) / 1000.0

    def snapshot(self) -> PowerSnapshot:
        """
        Returns a point-in-time snapshot without calling into NVML.

        NVML reads happen only in the background sampling thread.
        """
        now = time.perf_counter()
        with self._lock:
            samples = list(self._samples)
            joules_total = float(self._joules_total)
            last_t_s = self._last_t_s
            last_watts = self._last_watts
            last_temp_c = self._last_temp_c

        watts = float(last_watts) if last_watts is not None else 0.0
        t_s = float(last_t_s) if last_t_s is not None else now
        temp_c = float(last_temp_c) if last_temp_c is not None else 0.0
        watts_avg = self._average_watts_from_samples(samples)

        return PowerSnapshot(t_s=t_s, watts=watts, watts_avg=watts_avg, joules_total=joules_total, temperature_c=temp_c)

    def _average_watts_from_samples(self, samples: list[tuple[float, float]]) -> float:
        if not samples:
            return 0.0
        if len(samples) == 1:
            return float(samples[0][1])

        # Time-weighted average over the sample span using trapezoidal integration.
        joules = 0.0
        for i in range(1, len(samples)):
            t0, w0 = samples[i - 1]
            t1, w1 = samples[i]
            dt = t1 - t0
            if dt <= 0.0:
                continue
            joules += 0.5 * (w0 + w1) * dt

        dt_total = samples[-1][0] - samples[0][0]
        if dt_total <= 1e-9:
            return float(samples[-1][1])
        return joules / dt_total

    def _run(self) -> None:
        # Sampling loop; runs independently of the inference loop.
        while not self._stop.is_set():
            t_s = time.perf_counter()
            try:
                watts = self._read_instant_watts()
                temp_c = float(
                    pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[arg-type]
                )
            except Exception:
                time.sleep(self.sample_interval_s)
                continue

            with self._lock:
                if self._last_t_s is not None and self._last_watts is not None:
                    dt = t_s - self._last_t_s
                    if dt > 0.0:
                        self._joules_total += 0.5 * (self._last_watts + watts) * dt

                self._last_t_s = t_s
                self._last_watts = watts
                self._last_temp_c = temp_c
                self._samples.append((t_s, watts))

                cutoff = t_s - self.window_s
                while self._samples and self._samples[0][0] < cutoff:
                    self._samples.popleft()

            time.sleep(self.sample_interval_s)

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# 1) SYSTEM STATE (MANDATORY)
@dataclass(frozen=True)
class SystemState:
    # NOTE: Field names intentionally match the required state-vector keys.
    T: float  # gpu_temperature_c (°C)
    P: float  # gpu_power_watts (W)
    tau: float  # tokens_per_second (TPS)
    N: int  # active_decodes (integer)
    Q: float  # kv_cache_pressure (0–1)


# 2) POWER MODEL (EQUATION — MUST IMPLEMENT)
def power_model(tau, N, Q, P_idle, alpha, beta, gamma):
    return P_idle + alpha * tau + beta * N + gamma * Q


# 3) THERMAL DYNAMICS (RC MODEL — REQUIRED)
def thermal_step(T, P, dt, C, R, T_amb):
    return T + dt * ((P / C) - ((T - T_amb) / (R * C)))


# 4) TOKEN THROUGHPUT MODEL (CORE INSIGHT)
def throughput_model(N, tau_single, Q, lam, mu):
    return (N * tau_single) / (1.0 + lam * (N - 1) ** 2 + mu * Q)


# 5) KV-CACHE PRESSURE MODEL
def kv_pressure(N, ctx_len, ctx_max):
    return min(1.0, (N * ctx_len) / ctx_max)


# 6) OBJECTIVE FUNCTION (PHYSICS-AWARE)
def objective(tau, P, T, Q, T_safe, lP, lT, lQ):
    return tau - lP * P - lT * max(0.0, T - T_safe) - lQ * Q


# 7) CONTROL LAW (NO RL — PURE CONTROL THEORY)
def control_step(N, tau, tau_target, P, P_budget, T, T_safe, k1, k2, k3):
    dN = k1 * (tau_target - tau) - k2 * (P - P_budget) - k3 * (T - T_safe)
    return max(1, int(round(N + dN)))


# 8) MULTI-DECODE TIME-DIVISION SCHEDULING
def duty_cycle(weights):
    total = sum(weights)
    return [w / total for w in weights]


class SmoothWRRScheduler:
    """
    Deterministic weighted duty-cycling scheduler (Smooth Weighted Round Robin).

    - Activates a *subset* of decodes per step (default: one).
    - Rotates fairly according to weights.
    - Deterministic (no randomness) to avoid control noise.
    """

    def __init__(self, weights: List[float]) -> None:
        w = duty_cycle(weights)
        self._w = w
        self._cur = [0.0 for _ in w]

    def next_index(self) -> int:
        # Smooth weighted round robin:
        #   cur[i] += w[i]
        #   pick argmax(cur)
        #   cur[pick] -= 1
        best = 0
        best_val = None
        for i, wi in enumerate(self._w):
            self._cur[i] += wi
            v = self._cur[i]
            if best_val is None or v > best_val:
                best = i
                best_val = v
        self._cur[best] -= 1.0
        return best


@dataclass
class PhysicsConfig:
    # Control/update rate
    dt_s: float = 0.1  # ~10 Hz

    # Hard constraints
    P_budget_w: float = 110.0
    T_safe_c: float = 80.0

    # Target throughput
    tau_target_tps: float = 800.0

    # Power model parameters
    P_idle_w: float = 20.0
    alpha_w_per_tps: float = 0.10
    beta_w_per_decode: float = 2.0
    gamma_w_per_Q: float = 10.0

    # Thermal model parameters (RC)
    C_j_per_c: float = 200.0
    R_c_per_w: float = 0.40
    T_amb_c: float = 25.0

    # Throughput contention model parameters
    tau_single_tps: float = 800.0
    lam: float = 0.10
    mu: float = 0.50

    # Objective weights
    lP: float = 0.0
    lT: float = 0.0
    lQ: float = 0.0

    # Control gains (units-normalized); tune via system identification
    k1: float = 0.01
    k2: float = 0.05
    k3: float = 0.05


@dataclass(frozen=True)
class PhysicsStep:
    x: SystemState
    tau_pred: float
    P_pred: float
    T_next_pred: float
    J: float
    tau_setpoint: float
    N_next: int


class PhysicsController:
    """
    Physics-driven controller that maps measured metrics -> (N, target tau).

    Notes:
    - This implementation keeps the control law explainable and strictly based on
      the provided equations.
    - "N" is the commanded parallelism level; enforcement depends on the runtime.
    """

    def __init__(self, cfg: PhysicsConfig, *, ctx_max: int, prompt_tokens: int) -> None:
        self._cfg = cfg
        self._ctx_max = max(1, int(ctx_max))
        self._prompt_tokens = max(0, int(prompt_tokens))

        self._N = 1
        self._T_hat: float | None = None
        self._alpha_hat = max(1e-6, float(cfg.alpha_w_per_tps))

    @property
    def N(self) -> int:
        return self._N

    def step(
        self,
        *,
        T_meas_c: float,
        P_meas_w: float,
        tau_meas_tps: float,
        tokens_total: int,
        dt_s: float | None = None,
    ) -> PhysicsStep:
        c = self._cfg
        dt = float(c.dt_s if dt_s is None else dt_s)

        ctx_len = self._prompt_tokens + max(0, int(tokens_total))
        Q = kv_pressure(self._N, ctx_len, self._ctx_max)

        tau_pred = throughput_model(self._N, c.tau_single_tps, Q, c.lam, c.mu)
        P_pred = power_model(
            tau_pred, self._N, Q, c.P_idle_w, self._alpha_hat, c.beta_w_per_decode, c.gamma_w_per_Q
        )

        if self._T_hat is None:
            self._T_hat = float(T_meas_c)
        self._T_hat = thermal_step(self._T_hat, float(P_meas_w), dt, c.C_j_per_c, c.R_c_per_w, c.T_amb_c)
        T_next_pred = self._T_hat

        x = SystemState(T=float(T_meas_c), P=float(P_meas_w), tau=float(tau_meas_tps), N=int(self._N), Q=float(Q))
        J = objective(x.tau, x.P, x.T, x.Q, c.T_safe_c, c.lP, c.lT, c.lQ)

        # Online power-model adaptation: estimate effective alpha from measurements
        # using the provided linear power model structure. This improves hard-cap
        # enforcement without introducing any black-box tuning.
        if x.tau > 1e-6:
            alpha_inst = (x.P - c.P_idle_w - c.beta_w_per_decode * self._N - c.gamma_w_per_Q * Q) / x.tau
            if alpha_inst > 1e-6:
                eta = 0.05
                self._alpha_hat = (1.0 - eta) * self._alpha_hat + eta * float(alpha_inst)
                self._alpha_hat = float(min(10.0, max(1e-6, self._alpha_hat)))

        N_next = control_step(
            self._N,
            x.tau,
            c.tau_target_tps,
            x.P,
            c.P_budget_w,
            x.T,
            c.T_safe_c,
            c.k1,
            c.k2,
            c.k3,
        )

        # Safety: never increase parallelism while violating hard constraints.
        if x.P > c.P_budget_w or x.T > c.T_safe_c:
            N_next = min(N_next, self._N)

        N_next = max(1, int(N_next))

        Q_next = kv_pressure(N_next, ctx_len, self._ctx_max)

        # Hard constraint enforcement via physics: compute a tau setpoint that
        # respects power and (predicted) thermal limits.
        tau_limits = [float(c.tau_target_tps)]

        if self._alpha_hat > 0.0:
            tau_power = (
                c.P_budget_w - c.P_idle_w - c.beta_w_per_decode * N_next - c.gamma_w_per_Q * Q_next
            ) / self._alpha_hat
            tau_limits.append(tau_power)

        # Thermal: constrain power so T_{k+1} <= T_safe, then convert to tau via power model.
        if dt > 0.0 and c.R_c_per_w > 0.0 and self._alpha_hat > 0.0:
            P_thermal_limit = c.C_j_per_c * (c.T_safe_c - x.T) / dt + (x.T - c.T_amb_c) / c.R_c_per_w
            tau_thermal = (
                P_thermal_limit - c.P_idle_w - c.beta_w_per_decode * N_next - c.gamma_w_per_Q * Q_next
            ) / self._alpha_hat
            tau_limits.append(tau_thermal)

        tau_setpoint = max(1.0, min(tau_limits))

        self._N = N_next

        return PhysicsStep(
            x=x,
            tau_pred=float(tau_pred),
            P_pred=float(P_pred),
            T_next_pred=float(T_next_pred),
            J=float(J),
            tau_setpoint=float(tau_setpoint),
            N_next=int(N_next),
        )

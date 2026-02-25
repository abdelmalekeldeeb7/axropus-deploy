from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ControllerOutput:
    next_N: int
    predicted_tps: float
    predicted_power: float
    objective_value: float


class PhysicsController:
    """
    Deterministic, side-effect free (no IO) physics-based controller.

    The controller tracks the state variables:
      T  : temperature (°C)
      P  : power (W)
      tau: throughput (tokens/sec)
      N  : active decode streams (integer)
      kv : KV-cache pressure (dimensionless)

    Physics models implemented exactly as specified:
      1) Power Model:
         P = alpha * tau + beta * N + gamma

      2) Thermal Dynamics (Discrete RC Model):
         T_{t+1} = T_t + (dt / C) * (P - ((T_t - T_amb) / R))

      3) Throughput Saturation:
         tau = tau_max * (1 - exp(-lambda_ * N))

      4) KV Cache Pressure:
         kv = (N * tau) / ctx_limit

      5) Objective Function:
         J = k1 * tau - k2 * P - k3 * kv

    Control law:
      - Enumerate N candidates in [1, N_max]
      - Predict (tau, P, kv, T_next) for each candidate
      - Enforce constraints P <= P_max and T_next <= T_max
      - Choose the candidate that maximizes J
      - Penalize large N jumps (oscillation suppression) using an augmented score:
          score = J - abs(k2 * beta) * abs(N - N_current)
    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        gamma: float,
        C: float,
        R: float,
        T_amb: float,
        tau_max: float,
        lambda_: float,
        ctx_limit: float,
        k1: float,
        k2: float,
        k3: float,
        P_max: float,
        T_max: float,
        N_max: int,
    ) -> None:
        if C <= 0.0:
            raise ValueError("C must be > 0")
        if R <= 0.0:
            raise ValueError("R must be > 0")
        if ctx_limit <= 0.0:
            raise ValueError("ctx_limit must be > 0")
        if N_max < 1:
            raise ValueError("N_max must be >= 1")

        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)

        self._C = float(C)
        self._R = float(R)
        self._T_amb = float(T_amb)

        self._tau_max = float(tau_max)
        self._lambda = float(lambda_)
        self._ctx_limit = float(ctx_limit)

        self._k1 = float(k1)
        self._k2 = float(k2)
        self._k3 = float(k3)

        self._P_max = float(P_max)
        self._T_max = float(T_max)
        self._N_max = int(N_max)

        self._T = 0.0
        self._P = 0.0
        self._tau = 0.0
        self._N = 1
        self._kv = 0.0

    @property
    def T(self) -> float:
        return self._T

    @property
    def P(self) -> float:
        return self._P

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def N(self) -> int:
        return self._N

    @property
    def kv(self) -> float:
        return self._kv

    def _power(self, tau: float, N: int) -> float:
        return self._alpha * tau + self._beta * float(N) + self._gamma

    def _thermal_next(self, T: float, P: float, dt: float) -> float:
        dt_eff = 0.0 if not math.isfinite(dt) or dt <= 0.0 else float(dt)
        return T + (dt_eff / self._C) * (P - ((T - self._T_amb) / self._R))

    def _throughput(self, N: int) -> float:
        N_eff = max(1, min(int(N), self._N_max))
        return self._tau_max * (1.0 - math.exp(-self._lambda * float(N_eff)))

    def _kv_pressure(self, N: int, tau: float) -> float:
        return (float(N) * tau) / self._ctx_limit

    def _objective(self, tau: float, P: float, kv: float) -> float:
        return self._k1 * tau - self._k2 * P - self._k3 * kv

    def predict(self, N: int, tps: float) -> Tuple[float, float]:
        """
        Predict throughput and power for a candidate parallelism N.

        Args:
            N: Candidate number of active decode streams.
            tps: Current measured throughput (accepted for API compatibility).

        Returns:
            (predicted_tps, predicted_power)
        """
        N_eff = max(1, min(int(N), self._N_max))
        tau_pred = self._throughput(N_eff)
        P_pred = self._power(tau_pred, N_eff)
        return float(tau_pred), float(P_pred)

    def step(self, power_watts: float, temp_c: float, tps: float, dt: float) -> Tuple[int, float, float, float]:
        """
        Advance the controller by one discrete-time step.

        Inputs:
            power_watts: Measured GPU power (W)
            temp_c: Measured GPU temperature (°C)
            tps: Measured throughput (tokens/sec)
            dt: Timestep (sec)

        Outputs:
            (next_N, predicted_tps, predicted_power, objective_value)
        """
        T_meas = float(temp_c)
        P_meas = float(power_watts)
        tau_meas = float(tps)

        self._T = T_meas
        self._P = P_meas
        self._tau = tau_meas
        self._kv = self._kv_pressure(self._N, tau_meas)

        best_N = 1
        best_tau = self._throughput(best_N)
        best_P = self._power(best_tau, best_N)
        best_J = self._objective(best_tau, best_P, self._kv_pressure(best_N, best_tau))
        best_score = -math.inf

        jump_weight = abs(self._k2 * self._beta)

        for cand_N in range(1, self._N_max + 1):
            tau_pred = self._throughput(cand_N)
            P_pred = self._power(tau_pred, cand_N)
            kv_pred = self._kv_pressure(cand_N, tau_pred)
            T_pred = self._thermal_next(T_meas, P_pred, dt)

            if P_pred > self._P_max or T_pred > self._T_max:
                continue

            J = self._objective(tau_pred, P_pred, kv_pred)
            score = J - jump_weight * abs(cand_N - self._N)

            if score > best_score:
                best_score = score
                best_N = cand_N
                best_tau = tau_pred
                best_P = P_pred
                best_J = J

        next_N = max(1, min(int(best_N), self._N_max))
        self._N = next_N

        return int(next_N), float(best_tau), float(best_P), float(best_J)

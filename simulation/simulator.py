# nav_mpc/simulation/simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import sympy as sp

from core.models.dynamics import SystemModel


@dataclass
class SimulatorConfig:
    dt: float                      # control / integration step
    method: Literal["euler", "rk4"] = "rk4"
    substeps: int = 10              # if you want dt/substeps internal steps


class ContinuousSimulator:
    """
    Generic continuous-time simulator:

        x_dot = f(x, u)

    built from a SystemModel's symbolic dynamics.

    Provides:
      - step(x, u): one integration step
      - rollout(x0, U): simulate over a sequence of controls
    """

    def __init__(self, system: SystemModel, config: SimulatorConfig) -> None:
        self.system = system
        self.config = config
        self.nx = system.state_dim
        self.nu = system.input_dim

        # Build numeric f(x,u) from symbolic dynamics
        x_sym = system.state_symbolic()
        u_sym = system.input_symbolic()
        f_sym = system.dynamics_symbolic()

        # lambdify: (x, u) -> f(x,u)
        # x and u will be 1D numpy arrays
        self._f_fun = sp.lambdify((x_sym, u_sym), f_sym, "numpy")

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Evaluate f(x, u) numerically.
        """
        x = np.asarray(x, dtype=float).reshape(self.nx)
        u = np.asarray(u, dtype=float).reshape(self.nu)

        f_val = self._f_fun(x, u)      # may be list/array/matrix
        f_val = np.asarray(f_val, dtype=float).reshape(self.nx)
        return f_val

    def _euler_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        return x + dt * self.f(x, u)

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Classic 4th-order Runge-Kutta.
        """
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Integrate from x with constant u over one control interval config.dt.
        If substeps > 1, splits dt into smaller steps.
        """
        dt = self.config.dt
        n_sub = max(1, int(self.config.substeps))
        h = dt / n_sub

        xk = np.asarray(x, dtype=float).reshape(self.nx)

        for _ in range(n_sub):
            if self.config.method == "euler":
                xk = self._euler_step(xk, u, h)
            elif self.config.method == "rk4":
                xk = self._rk4_step(xk, u, h)
            else:
                raise ValueError(f"Unknown integration method: {self.config.method}")

        return xk
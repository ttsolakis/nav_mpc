# nav_mpc/models/unicycle_kinematic_model.py

import sympy as sp
from models.dynamics import SystemModel


class UnicycleKinematicModel(SystemModel):
    """
    Planar kinematic model with body-velocity states and wheel-acceleration inputs.

    State:
        x = [px, py, phi, v, r]^T
            v: forward (body) speed [m/s]
            r: yaw rate [rad/s]

    Input:
        u = [alpha_l, alpha_r]^T
            alpha_l: left wheel angular accel [rad/s^2]
            alpha_r: right wheel angular accel [rad/s^2]

    Dynamics:
        px_dot  = v*cos(phi)
        py_dot  = v*sin(phi)
        phi_dot = r
        v_dot   = 0.5*R*(alpha_l + alpha_r)
        r_dot   = 0.5*(R/L)*(alpha_r - alpha_l)

    Notes:
    - R is wheel radius [m]
    - L is half-track [m]
    """

    def __init__(self, *, R: float = 0.040, L: float = 0.062) -> None:
        self.R = float(R)
        self.L = float(L)
        super().__init__(state_dim=5, input_dim=2)

    def build_dynamics(self) -> None:
        x = self.x_sym
        u = self.u_sym

        px, py, phi, v, r = x[0], x[1], x[2], x[3], x[4]
        alpha_l, alpha_r = u[0], u[1]

        R = sp.Float(self.R)
        L = sp.Float(self.L)

        px_dot = v * sp.cos(phi)
        py_dot = v * sp.sin(phi)
        phi_dot = r

        v_dot = sp.Rational(1, 2) * R * (alpha_l + alpha_r)
        r_dot = sp.Rational(1, 2) * (R / L) * (alpha_r - alpha_l)

        self.f_sym = sp.Matrix([px_dot, py_dot, phi_dot, v_dot, r_dot])

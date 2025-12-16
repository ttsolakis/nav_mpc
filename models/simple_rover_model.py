import sympy as sp
from models.dynamics import SystemModel


class SimpleRoverModel(SystemModel):
    """
    Rover kinematic model with input-rate formulation.

    State:
        x = [px, py, phi, omega_l, omega_r]^T

    Input:
        u = [alpha_l, alpha_r]^T  where alpha = d/dt omega
    """

    def __init__(self) -> None:
        state_dim = 5
        input_dim = 2

        self.R = 0.040  # wheel radius [m]
        self.L = 0.062  # half track  [m]

        super().__init__(state_dim, input_dim)

    def build_dynamics(self) -> None:
        x = self.x_sym
        u = self.u_sym

        px, py, phi = x[0], x[1], x[2]
        omega_l, omega_r = x[3], x[4]

        alpha_l, alpha_r = u[0], u[1]

        R = self.R
        L = self.L

        v = sp.Rational(1, 2) * R * (omega_r + omega_l)
        r = sp.Rational(1, 2) * R / L * (omega_r - omega_l)

        px_dot  = v * sp.cos(phi)
        py_dot  = v * sp.sin(phi)
        phi_dot = r

        omega_l_dot = alpha_l
        omega_r_dot = alpha_r

        self.f_sym = sp.Matrix([px_dot, py_dot, phi_dot, omega_l_dot, omega_r_dot])

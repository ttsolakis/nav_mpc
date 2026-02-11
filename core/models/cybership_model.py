# nav_mpc/models/cybership_model.py

import sympy as sp
from core.models.dynamics import SystemModel


class CybershipModel(SystemModel):
    """
    Cybership 3-DOF planar marine craft with actuator-thrust mapping.

    State:
        x = [px, py, psi, ux, uy, r]^T
            (px, py, psi): position/heading in {n}
            (ux, uy, r): body-fixed velocities in {b}

    Input:
        u_in = [thrust_left, thrust_right, thrust_bow, azimuth_left, azimuth_right]^T
            thrust_*:  force magnitudes [N]
            azimuth_*: angles [rad] (body frame)

    Dynamics:
        eta_dot = R(psi) * v
        v_dot   = M^{-1} * (tau - (C + D) * v)

    where tau is computed from the thruster configuration.

    """

    def __init__(
        self,
        *,
        # Geometry
        l: float = 1.255,
        b: float = 0.29,
        xg: float = 0.046,
        # Actuator configuration
        l_l: float = 0.5,
        l_r: float = 0.5,
        l_b: float = 0.5,
        b_l: float = 0.14,
        b_r: float = 0.14,
        # Mass properties
        m: float = 23.8,
        Iz: float = 1.76,
        # Hydro on/off (keep as scalar multiplier like in your code)
        hydro_effects: float = 1.0,
    ) -> None:
        # Store parameters
        self.l = float(l)
        self.b = float(b)
        self.xg = float(xg)

        self.l_l = float(l_l)
        self.l_r = float(l_r)
        self.l_b = float(l_b)
        self.b_l = float(b_l)
        self.b_r = float(b_r)

        self.m = float(m)
        self.Iz = float(Iz)

        self.hydro_effects = float(hydro_effects)

        super().__init__(state_dim=6, input_dim=5)

    def build_dynamics(self) -> None:
        x = self.x_sym
        u_in = self.u_sym

        # ===== State =====
        px, py, psi, ux, uy, r = x[0], x[1], x[2], x[3], x[4], x[5]
        v = sp.Matrix([ux, uy, r])

        # ===== Input (thrusters) =====
        thrust_left, thrust_right, thrust_bow, az_left, az_right = (
            u_in[0],
            u_in[1],
            u_in[2],
            u_in[3],
            u_in[4],
        )

        # ===== Parameters (floats for stable codegen) =====
        m = sp.Float(self.m)
        Iz = sp.Float(self.Iz)
        xg = sp.Float(self.xg)

        l_l = sp.Float(self.l_l)
        l_r = sp.Float(self.l_r)
        l_b = sp.Float(self.l_b)
        b_l = sp.Float(self.b_l)
        b_r = sp.Float(self.b_r)

        hydro = sp.Float(self.hydro_effects)

        # ===== Hydrodynamic coefficients (Skjetne2004) =====
        Xud, Yvd, Yrd, Nvd, Nrd = -2.0, -10.0, 0.0, 0.0, -1.0
        Xu, Xuu, Xuuu = -0.72253, -1.32742, -5.86643
        Yv, Yvv, Yrv, Yr, Yvr, Yrr = -0.88965, -36.47287, -0.805, -7.25, -0.845, -3.45
        Nv, Nvv, Nrv, Nr, Nvr, Nrr = 0.0313, 3.95645, 0.13, -1.9, 0.08, -0.75

        Xud = sp.Float(Xud)
        Yvd = sp.Float(Yvd)
        Yrd = sp.Float(Yrd)
        Nvd = sp.Float(Nvd)
        Nrd = sp.Float(Nrd)

        Xu = sp.Float(Xu)
        Xuu = sp.Float(Xuu)
        Xuuu = sp.Float(Xuuu)

        Yv = sp.Float(Yv)
        Yvv = sp.Float(Yvv)
        Yrv = sp.Float(Yrv)
        Yr = sp.Float(Yr)
        Yvr = sp.Float(Yvr)
        Yrr = sp.Float(Yrr)

        Nv = sp.Float(Nv)
        Nvv = sp.Float(Nvv)
        Nrv = sp.Float(Nrv)
        Nr = sp.Float(Nr)
        Nvr = sp.Float(Nvr)
        Nrr = sp.Float(Nrr)

        # ===== Rotation matrix R(psi) =====
        R_psi = sp.Matrix(
            [
                [sp.cos(psi), -sp.sin(psi), 0],
                [sp.sin(psi),  sp.cos(psi), 0],
                [0, 0, 1],
            ]
        )

        # ===== Mass matrix M_RB (Eq.3 pg.5) =====
        M_RB = sp.Matrix(
            [
                [m, 0, 0],
                [0, m, m * xg],
                [0, m * xg, Iz],
            ]
        )

        # ===== Coriolis matrix C_RB (Eq.4 pg.6) =====
        C_RB = sp.Matrix(
            [
                [0, 0, -m * (xg * r + uy)],
                [0, 0,  m * ux],
                [m * (xg * r + uy), -m * ux, 0],
            ]
        )

        # ===== Added mass matrix M_A (Eq.6.98 Fossen2021) =====
        M_A = hydro * sp.Matrix(
            [
                [-Xud, 0, 0],
                [0, -Yvd, -Yrd],
                [0, -Nvd, -Nrd],
            ]
        )

        # ===== Added Coriolis matrix C_A (Eq.9 Fossen2021 form) =====

        c13 = Yvd * uy + Yrd * r
        c23 = -Xud * ux
        C_A = hydro * sp.Matrix(
            [
                [0, 0, c13],
                [0, 0, c23],
                [-c13, -c23, 0],
            ]
        )

        # ===== Linear damping D_L (Eq.13) =====
        D_L = hydro * sp.Matrix(
            [
                [-Xu, 0, 0],
                [0, -Yv, -Yr],
                [0, -Nv, -Nr],
            ]
        )

        # ===== Nonlinear damping D_NL (Eq.13) =====
        d11 = -Xuu * sp.Abs(ux) - Xuuu * (ux**2)
        d22 = -Yvv * sp.Abs(uy) + Yrv * sp.Abs(r)
        d23 = -Yvr * sp.Abs(uy) - Yrr * sp.Abs(r)
        d32 = -Nvv * sp.Abs(uy) - Nrv * sp.Abs(r)
        d33 = -Nvr * sp.Abs(uy) - Nrr * sp.Abs(r)

        D_NL = hydro * sp.Matrix(
            [
                [d11, 0, 0],
                [0, d22, d23],
                [0, d32, d33],
            ]
        )

        # ===== Total matrices =====
        M = M_RB + M_A
        C = C_RB + C_A
        D = D_L + D_NL

        # ===== Thruster -> generalized forces (tau) (exactly like your numpy code) =====
        tau_x = thrust_left * sp.cos(az_left) + thrust_right * sp.cos(az_right)
        tau_y = thrust_left * sp.sin(az_left) + thrust_right * sp.sin(az_right) + thrust_bow

        tau_n = (
            - b_l * thrust_left * sp.cos(az_left)
            + b_r * thrust_right * sp.cos(az_right)
            - l_l * thrust_left * sp.sin(az_left)
            - l_r * thrust_right * sp.sin(az_right)
            + l_b * thrust_bow
        )

        tau = sp.Matrix([tau_x, tau_y, tau_n])

        # ===== Continuous dynamics =====
        eta_dot = R_psi * v
        v_dot = M.LUsolve(tau - (C + D) * v)  # use LUsolve for better numerical stability in codegen instead of explicit inverse
        # v_dot = M.inv() * (tau - (C + D) * v)


        self.f_sym = sp.Matrix([eta_dot[0], eta_dot[1], eta_dot[2], v_dot[0], v_dot[1], v_dot[2]])

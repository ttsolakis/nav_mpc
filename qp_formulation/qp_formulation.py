# nav_mpc/qp_formulation/qp_formulation.py

import sympy as sp
from models.dynamics import SystemModel

SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _subscript_index(i: int) -> str:
    return str(i).translate(SUBSCRIPT_DIGITS)


def build_linearized_system(system: SystemModel):
    """
    Construct symbolic continuous-time affine linearization

        xdot = Ā(x̄,ū) x + B̄(x̄,ū) u + c̄(x̄,ū),

    and also return fast NumPy callables:
        A_fun(x_bar, u_bar), B_fun(x_bar, u_bar), c_fun(x_bar, u_bar).
    """
    x_sym = system.state_symbolic()
    u_sym = system.input_symbolic()
    f_sym = system.dynamics_symbolic()

    A_sym = f_sym.jacobian(x_sym)
    B_sym = f_sym.jacobian(u_sym)

    # Operating point symbols x̄, ū
    x_bar_base = "x" + "\u0304"
    u_bar_base = "u" + "\u0304"

    x_bar_syms = [
        sp.symbols(f"{x_bar_base}{_subscript_index(i)}", real=True)
        for i in range(system.state_dim)
    ]
    u_bar_syms = [
        sp.symbols(f"{u_bar_base}{_subscript_index(j)}", real=True)
        for j in range(system.input_dim)
    ]

    x_bar_sym = sp.Matrix(x_bar_syms)
    u_bar_sym = sp.Matrix(u_bar_syms)

    # Substitute (x,u) -> (x̄,ū)
    subs_bar = {
        x_sym[i]: x_bar_sym[i] for i in range(system.state_dim)
    } | {
        u_sym[j]: u_bar_sym[j] for j in range(system.input_dim)
    }

    f_bar_sym = f_sym.subs(subs_bar)
    A_bar_sym = A_sym.subs(subs_bar)
    B_bar_sym = B_sym.subs(subs_bar)
    c_bar_sym = f_bar_sym - A_bar_sym * x_bar_sym - B_bar_sym * u_bar_sym

    # Full affine vector field (for debug only)
    xdot_lin_sym = A_bar_sym * x_sym + B_bar_sym * u_sym + c_bar_sym

    print("\nLinearized dynamics ẋ ≈")
    sp.pprint(xdot_lin_sym)

    # ---------- Lambdify for fast numeric evaluation ----------
    # Arguments will be (x_bar_0, ..., x_bar_{n-1}, u_bar_0, ..., u_bar_{m-1})
    bar_vars = list(x_bar_sym) + list(u_bar_sym)

    A_fun = sp.lambdify(bar_vars, A_bar_sym, "numpy")
    B_fun = sp.lambdify(bar_vars, B_bar_sym, "numpy")
    c_fun = sp.lambdify(bar_vars, c_bar_sym, "numpy")

    return A_fun, B_fun, c_fun

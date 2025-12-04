# nav_mpc/qp_formulation/qp_formulation.py

import sympy as sp
from models.dynamics import SystemModel

# Small helper just for pretty subscripts in x̄₀, x̄₁, ...
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _subscript_index(i: int) -> str:
    """Convert integer i to a string of Unicode subscript digits."""
    return str(i).translate(SUBSCRIPT_DIGITS)


def build_linearized_system(system: SystemModel):
    """
    Given a symbolic SystemModel with f(x,u), construct the symbolic
    continuous-time affine linearization:

        ẋ ≈ A(x̄,ū) x + B(x̄,ū) u + c(x̄,ū)

    where (x̄, ū) are symbolic operating-point variables (to be substituted).

    Parameters
    ----------
    system : SystemModel
        A model that provides:
          - state_dim, input_dim
          - state_symbolic() -> sympy Matrix (x)
          - input_symbolic() -> sympy Matrix (u)
          - dynamics_symbolic() -> sympy Matrix f(x,u)

    Returns
    -------
    A_bar_sym : sympy.Matrix
        Jacobian df/dx evaluated at (x̄,ū), i.e. A(x̄,ū).
    B_bar_sym : sympy.Matrix
        Jacobian df/du evaluated at (x̄,ū), i.e. B(x̄,ū).
    c_bar_sym : sympy.Matrix
        Affine term c(x̄,ū) = f(x̄,ū) - A(x̄,ū)x̄ - B(x̄,ū)ū.
    x_bar_sym : sympy.Matrix
        Symbolic operating-point state [x̄₀, x̄₁, ...]^T.
    u_bar_sym : sympy.Matrix
        Symbolic operating-point input [ū₀, ū₁, ...]^T.
    xdot_lin_sym : sympy.Matrix
        Full linearized vector field:
            ẋ = A_bar_sym * x + B_bar_sym * u + c_bar_sym
        expressed in terms of x, u, x̄, ū.
    """
    # Symbolic state, input, dynamics
    x_sym = system.state_symbolic()
    u_sym = system.input_symbolic()
    f_sym = system.dynamics_symbolic()

    # Jacobians df/dx, df/du
    A_sym = f_sym.jacobian(x_sym)
    B_sym = f_sym.jacobian(u_sym)

    # ------------------------------------------------------------------
    # Symbolic operating point (x̄, ū)
    # ------------------------------------------------------------------
    x_bar_base = "x" + "\u0304"  # "x̄"
    u_bar_base = "u" + "\u0304"  # "ū"

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

    # Substitute (x,u) -> (x̄,ū) in f, A, B
    subs_bar = {
        x_sym[i]: x_bar_sym[i] for i in range(system.state_dim)
    } | {
        u_sym[j]: u_bar_sym[j] for j in range(system.input_dim)
    }

    f_bar_sym = f_sym.subs(subs_bar)                                       # f(x̄, ū)
    A_bar_sym = A_sym.subs(subs_bar)                                       # A(x̄, ū)
    B_bar_sym = B_sym.subs(subs_bar)                                       # B(x̄, ū)
    c_bar_sym = f_bar_sym - A_bar_sym * x_bar_sym - B_bar_sym * u_bar_sym  # c(x̄, ū)

    # ------------------------------------------------------------------
    # Build full symbolic linearized (affine) model:
    #   ẋ = f(x̄,ū) + A(x̄,ū)(x - x̄) + B(x̄,ū)(u - ū) =>
    #   ẋ = A(x̄,ū)*x + B(x̄,ū)*u + c(x̄,ū) 
    # ------------------------------------------------------------------
    xdot_lin_sym = A_bar_sym * x_sym + B_bar_sym * u_sym + c_bar_sym
    
    print("\nLinearized dynamics ẋ ≈")
    sp.pprint(xdot_lin_sym)

    return A_bar_sym, B_bar_sym, c_bar_sym, x_bar_sym, u_bar_sym, xdot_lin_sym

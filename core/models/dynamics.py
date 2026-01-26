# nav_mpc/models/dynamics.py

from abc import ABC, abstractmethod
import sympy as sp


class SystemModel(ABC):
    """
    Abstract base class for continuous-time system models in symbolic form.

    Each concrete system must:
      - set state_dim, input_dim
      - use the provided symbolic state x_sym and input u_sym
      - define f_sym(x,u) via build_dynamics()

    Attributes
    ----------
    state_dim : int
        Dimension of the state vector x.
    input_dim : int
        Dimension of the input vector u.
    x_sym : sympy.Matrix
        Symbolic state vector (column) of length state_dim.
    u_sym : sympy.Matrix
        Symbolic input vector (column) of length input_dim.
    f_sym : sympy.Matrix
        Symbolic dynamics x_dot = f_sym(x_sym, u_sym).
    """

    def __init__(self, state_dim: int, input_dim: int) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Create generic symbolic state and input:
        # x = [x0, x1, ..., x_{nx-1}]^T
        # u = [u0, u1, ..., u_{nu-1}]^T
        x_symbols = sp.symbols(f"x0:{state_dim}", real=True)
        u_symbols = sp.symbols(f"u0:{input_dim}", real=True)

        self.x_sym = sp.Matrix(x_symbols)
        self.u_sym = sp.Matrix(u_symbols)

        # Placeholder for dynamics; child must set this in build_dynamics()
        self.f_sym: sp.Matrix | None = None

        # Let the child class actually define f_sym
        self.build_dynamics()

        # Basic sanity check
        if not isinstance(self.f_sym, sp.Matrix):
            raise TypeError("f_sym must be a sympy.Matrix.")
        if self.f_sym.shape != (self.state_dim, 1):
            raise ValueError(
                f"f_sym must have shape ({self.state_dim}, 1), "
                f"got {self.f_sym.shape}"
            )

    @abstractmethod
    def build_dynamics(self) -> None:
        """
        Child classes must set self.f_sym as a sympy.Matrix of shape (state_dim, 1),
        using self.x_sym and self.u_sym.
        """
        raise NotImplementedError

    # Convenience accessors (handy later in qp_formulation)
    def state_symbolic(self) -> sp.Matrix:
        return self.x_sym

    def input_symbolic(self) -> sp.Matrix:
        return self.u_sym

    def dynamics_symbolic(self) -> sp.Matrix:
        return self.f_sym

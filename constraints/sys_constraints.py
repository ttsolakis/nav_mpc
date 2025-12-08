# nav_mpc/constraints/sys_constraints.py

from abc import ABC, abstractmethod
import numpy as np
import sympy as sp


class SystemConstraints(ABC):
    """
    Abstract base class for system constraints.

    Responsibilities:
      - store numeric box bounds on x and u
      - let child classes define symbolic inequality constraints g(x,u) <= 0
        via build_system_constraints()
      - provide a constraints_symbolic() accessor, similar to dynamics_symbolic()
        in SystemModel.

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
    g_sym : sympy.Matrix
        Symbolic inequality constraints g(x,u) <= 0, shape (nc, 1).
    x_min, x_max : np.ndarray
        State box bounds, shape (state_dim,).
    u_min, u_max : np.ndarray or float
        Input box bounds.
    """

    def __init__(self, state_dim: int, input_dim: int) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Symbolic variables (parallel to SystemModel)
        x_symbols = sp.symbols(f"x0:{state_dim}", real=True)
        u_symbols = sp.symbols(f"u0:{input_dim}", real=True)

        self.x_sym = sp.Matrix(x_symbols)
        self.u_sym = sp.Matrix(u_symbols)

        # Default: no bounds (±∞)
        self.x_min = np.full(state_dim, -np.inf, dtype=float)
        self.x_max = np.full(state_dim,  np.inf, dtype=float)
        self.u_min = np.full(input_dim, -np.inf, dtype=float)
        self.u_max = np.full(input_dim,  np.inf, dtype=float)

        # Placeholder for constraints; child must define this in build_system_constraints()
        self._g_sym: sp.Matrix | None = None

    # ---------------------------
    # Symbolic inequality g(x,u) <= 0
    # ---------------------------
    @abstractmethod
    def build_system_constraints(self) -> sp.Matrix:
        """
        Child classes must construct and return g(x,u) as a sympy.Matrix of
        shape (nc, 1), using self.x_sym and self.u_sym (or system-provided
        symbols, if they prefer).

        Example:
            x = self.x_sym
            u = self.u_sym
            g1 = u[0] - umax
            g2 = -u[0] + umin
            return sp.Matrix([g1, g2])
        """
        raise NotImplementedError

    def constraints_symbolic(self) -> sp.Matrix:
        """
        Convenience accessor, analogous to SystemModel.dynamics_symbolic().
        Lazily builds g(x,u) once by calling build_system_constraints().
        """
        if self._g_sym is None:
            self._g_sym = self.build_system_constraints()

            if not isinstance(self._g_sym, sp.Matrix):
                raise TypeError("build_system_constraints() must return a sympy.Matrix.")
            if self._g_sym.shape[1] != 1:
                raise ValueError(
                    f"g_sym must be a column vector (nc, 1), got {self._g_sym.shape}"
                )

        return self._g_sym

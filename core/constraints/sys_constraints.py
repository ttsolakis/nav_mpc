from abc import ABC, abstractmethod
import numpy as np
import sympy as sp


class SystemConstraints(ABC):
    """
    Abstract base class for system constraints.
    ...
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
        self._nc: int | None = None    # number of constraints

    # ---------------------------
    # Symbolic inequality g(x,u) <= 0
    # ---------------------------
    @abstractmethod
    def build_system_constraints(self) -> sp.Matrix:
        ...
        raise NotImplementedError

    def constraints_symbolic(self) -> sp.Matrix:
        """
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

            # Store number of constraints
            self._nc = int(self._g_sym.shape[0])

        return self._g_sym

    @property
    def constraints_dim(self) -> int:
        """
        Number of inequality constraints nc.
        """
        if self._nc is None:
            # Ensure g_sym has been built
            _ = self.constraints_symbolic()
        return int(self._nc)

    def get_bounds(self):
        """
        Return numeric box bounds as (x_min, x_max, u_min, u_max).
        Copies are returned to avoid accidental in-place modification.
        """
        return (
            self.x_min.copy(),
            self.x_max.copy(),
            self.u_min.copy(),
            self.u_max.copy(),
        )

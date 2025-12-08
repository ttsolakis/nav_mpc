# nav_mpc/objectives/objectives.py

from abc import ABC, abstractmethod
import sympy as sp
from models.dynamics import SystemModel

class Objective(ABC):
    """
    Abstract base class for MPC stage cost definitions in symbolic form.

    Each concrete objective:
      - has access to system.state_symbolic(), system.input_symbolic()
      - defines symbolic error maps e_x(x - x_ref), e_u(u - u_ref)
      - stores numeric Q, QN, R, x_ref, u_ref for later use.
    """

    def __init__(self, system: SystemModel) -> None:
        self.system = system

        # Use the same symbols as the system (so everything is consistent)
        self.x_sym = system.state_symbolic()   # (nx, 1)
        self.u_sym = system.input_symbolic()   # (nu, 1)

    @abstractmethod
    def build_state_error(self) -> sp.Matrix:
        """
        Return e_x(x - x_ref) as a sympy column vector (nx, 1) or (n_ex, 1).
        """
        raise NotImplementedError

    @abstractmethod
    def build_input_error(self) -> sp.Matrix:
        """
        Return e_u(u - u_ref) as a sympy column vector (nu, 1) or (n_eu, 1).
        """
        raise NotImplementedError

    # Convenience accessors (like dynamics_symbolic / constraints_symbolic)
    def state_error_symbolic(self) -> sp.Matrix:
        return self.build_state_error()

    def input_error_symbolic(self) -> sp.Matrix:
        return self.build_input_error()

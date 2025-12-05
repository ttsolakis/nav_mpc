# nav_mpc/constraints/constraints.py

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SystemConstraints(ABC):
    """
    Abstract base class for simple box bounds:

        x_min <= x <= x_max
        u_min <= u <= u_max

    No symbolic stuff, no linearization here.
    Each system-specific subclass just sets these arrays.
    """

    def __init__(self, state_dim: int, input_dim: int) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Default: unconstrained
        self.x_min = -np.inf * np.ones(self.state_dim)
        self.x_max = +np.inf * np.ones(self.state_dim)
        self.u_min = -np.inf * np.ones(self.input_dim)
        self.u_max = +np.inf * np.ones(self.input_dim)

        # Let the child override them
        self.build_bounds()

        # Ensure correct shapes and types
        self.x_min = np.asarray(self.x_min, dtype=float).reshape(self.state_dim)
        self.x_max = np.asarray(self.x_max, dtype=float).reshape(self.state_dim)
        self.u_min = np.asarray(self.u_min, dtype=float).reshape(self.input_dim)
        self.u_max = np.asarray(self.u_max, dtype=float).reshape(self.input_dim)

    @abstractmethod
    def build_bounds(self) -> None:
        """
        Child must set x_min, x_max, u_min, u_max.
        If a bound is not set, it can remain ±inf.
        """
        raise NotImplementedError

    # Convenience accessor
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.x_min, self.x_max, self.u_min, self.u_max

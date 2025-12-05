# nav_mpc/objectives/objectives.py

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
class Objective(ABC):
    """
    Base class for quadratic MPC objectives of the form

        ℓ(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u

    (and optionally a terminal cost with QN).

    This class only stores Q, QN, R.
    """

    def __init__(self, state_dim: int, input_dim: int) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Default weights (child overrides in build_weights)
        self.Q = np.eye(self.state_dim)
        self.QN = np.eye(self.state_dim)
        self.R = np.eye(self.input_dim)

        self.build_weights()

        # Basic sanity checks
        self.Q = np.asarray(self.Q, dtype=float).reshape(self.state_dim, self.state_dim)
        self.QN = np.asarray(self.QN, dtype=float).reshape(self.state_dim, self.state_dim)
        self.R = np.asarray(self.R, dtype=float).reshape(self.input_dim, self.input_dim)

    @abstractmethod
    def build_weights(self) -> None:
        """
        Child must set self.Q, self.QN, self.R (numpy arrays with correct shapes).
        """
        raise NotImplementedError

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (Q, QN, R).
        """
        return self.Q, self.QN, self.R

# nav_mpc/objectives/objectives.py

from abc import ABC, abstractmethod
import numpy as np
from nav_mpc.models.dynamics import SystemModel


class Objective(ABC):
    """
    Abstract base class for a quadratic *stage* cost ℓ(x, u).

    The intended form is:

        ℓ(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u

    but this class only exposes the parameters (x_ref, Q, R).
    The MPC/QP layer will decide:
      - how many stages there are (horizon length),
      - at which stages this cost is applied.
    """

    def __init__(self, model: SystemModel) -> None:
        self.model = model
        self.state_dim = model.state_dim
        self.input_dim = model.input_dim

    @abstractmethod
    def get_x_ref(self) -> np.ndarray:
        """
        Return the reference state x_ref for this stage.

        Returns
        -------
        x_ref : np.ndarray
            Reference state, shape (state_dim,).
        """
        raise NotImplementedError

    @abstractmethod
    def get_Q(self) -> np.ndarray:
        """
        Return the state cost matrix Q for this stage.

        Returns
        -------
        Q : np.ndarray
            State cost matrix, shape (state_dim, state_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def get_R(self) -> np.ndarray:
        """
        Return the input cost matrix R for this stage.

        Returns
        -------
        R : np.ndarray
            Input cost matrix, shape (input_dim, input_dim).
        """
        raise NotImplementedError

# nav_mpc/objectives/waveshare_rover_set_point_objective.py

import numpy as np

from nav_mpc.objectives.objectives import Objective
from nav_mpc.models.waveshare_rover_model import WaveshareRoverModel


class WaveshareRoverSetPointObjective(Objective):
    """
    Set-point stage objective for the Waveshare rover:

        ℓ(x, u) = (x - x_goal)^T Q (x - x_goal) + u^T R u

    where x = [px, py, phi]^T.

    This class only defines the *stage* cost parameters (x_goal, Q, R).
    The MPC/QP layer will decide:
      - horizon length,
      - at which stages this cost is used (e.g. all stages, or only terminal).
    """

    def __init__(
        self,
        model: WaveshareRoverModel,
        x_goal: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        super().__init__(model)

        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        assert x_goal.size == self.state_dim, (
            f"x_goal must have size {self.state_dim}, got {x_goal.size}"
        )

        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)

        assert Q.shape == (self.state_dim, self.state_dim), (
            f"Q must have shape ({self.state_dim}, {self.state_dim}), got {Q.shape}"
        )
        assert R.shape == (self.input_dim, self.input_dim), (
            f"R must have shape ({self.input_dim}, {self.input_dim}), got {R.shape}"
        )

        self._x_goal = x_goal
        self._Q = Q
        self._R = R

    def get_x_ref(self) -> np.ndarray:
        """
        Return the desired state x_goal for this stage.
        """
        return self._x_goal

    def get_Q(self) -> np.ndarray:
        """
        Return the state cost matrix Q.
        """
        return self._Q

    def get_R(self) -> np.ndarray:
        """
        Return the input cost matrix R.
        """
        return self._R

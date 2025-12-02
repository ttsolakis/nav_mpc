# nav_mpc/planner/planner.py

import numpy as np
import tinympc


class TinyMPCPlanner:
    """
    Minimal TinyMPC-based planner with constant (A, B, Q, R).

    For now:
      - assumes fixed linear dynamics x_{k+1} = A x_k + B u_k
      - uses constant quadratic cost Q, R
      - no constraints
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        N: int,
        rho: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the TinyMPC solver with constant (A, B, Q, R).

        Parameters
        ----------
        A : np.ndarray
            State transition matrix, shape (nx, nx).
        B : np.ndarray
            Input matrix, shape (nx, nu).
        Q : np.ndarray
            State cost matrix, shape (nx, nx). (TinyMPC expects diagonal.)
        R : np.ndarray
            Input cost matrix, shape (nu, nu). (TinyMPC expects diagonal.)
        N : int
            Horizon length.
        rho : float, optional
            ADMM penalty parameter, by default 1.0.
        verbose : bool, optional
            If True, TinyMPC prints solver info.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)

        nx, nx2 = A.shape
        assert nx == nx2, f"A must be square, got {A.shape}"
        nx3, nu = B.shape
        assert nx3 == nx, f"A and B must have same number of rows, got {A.shape}, {B.shape}"
        assert Q.shape == (nx, nx), f"Q must have shape ({nx}, {nx}), got {Q.shape}"
        assert R.shape == (nu, nu), f"R must have shape ({nu}, {nu}), got {R.shape}"

        self.nx = nx
        self.nu = nu
        self.N = int(N)

        self.solver = tinympc.TinyMPC()
        self.solver.setup(A, B, Q, R, self.N, rho=rho, verbose=verbose)

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
    ):
        """
        Solve the MPC problem for the given initial state and references.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape (nx,).
        x_ref : np.ndarray, optional
            State reference trajectory, shape (nx, N).
            If None, zero reference is used.
        u_ref : np.ndarray, optional
            Input reference trajectory, shape (nu, N-1).
            If None, zero reference is used.

        Returns
        -------
        u0 : np.ndarray
            First control action, shape (nu,).
        X_pred : np.ndarray
            Predicted state trajectory, shape (N, nx).
        U_pred : np.ndarray
            Predicted input trajectory, shape (N-1, nu).
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        assert x0.size == self.nx, f"x0 must have size {self.nx}, got {x0.size}"

        # Default references: zero
        if x_ref is None:
            x_ref = np.zeros((self.nx, self.N), dtype=float)
        else:
            x_ref = np.asarray(x_ref, dtype=float)
            assert x_ref.shape == (self.nx, self.N), (
                f"x_ref must have shape ({self.nx}, {self.N}), got {x_ref.shape}"
            )

        if u_ref is None:
            u_ref = np.zeros((self.nu, self.N - 1), dtype=float)
        else:
            u_ref = np.asarray(u_ref, dtype=float)
            assert u_ref.shape == (self.nu, self.N - 1), (
                f"u_ref must have shape ({self.nu}, {self.N - 1}), got {u_ref.shape}"
            )

        self.solver.set_x0(x0)
        self.solver.set_x_ref(x_ref)
        self.solver.set_u_ref(u_ref)

        solution = self.solver.solve()

        u0 = solution["controls"]           # (nu,)
        X_pred = solution["states_all"].T   # (N, nx)
        U_pred = solution["controls_all"].T # (N-1, nu)

        return u0, X_pred, U_pred

# nav_mpc/planner/planner.py

import numpy as np
import scipy.sparse as sp
import osqp


class OSQPMPCPlanner:
    """
    Minimal OSQP-based MPC planner for LTI systems.

    System:
        x_{k+1} = A x_k + B u_k

    Cost:
        sum_{k=0}^{N-1} (x_k - x_ref)^T Q (x_k - x_ref)
        + (x_N - x_ref)^T Q (x_N - x_ref)
        + sum_{k=0}^{N-1} u_k^T R u_k

    For now:
      - fixed A, B, Q, R, horizon N
      - no explicit state/input bounds (only dynamics constraints)
      - x_ref is taken as constant over the horizon (first column of x_ref)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        N: int,
        verbose: bool = False,
    ) -> None:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(Q, dtype=float)
        R = np.asarray(R, dtype=float)

        nx, nx2 = A.shape
        assert nx == nx2, f"A must be square, got {A.shape}"
        nx3, nu = B.shape
        assert nx3 == nx, f"A and B must have same number of rows, got {A.shape}, {B.shape}"
        assert Q.shape == (nx, nx), f"Q must have shape ({nx}, {nx}), got {Q.shape}"
        assert R.shape == (nu, nu), f"R must have shape ({nu, nu}), got {R.shape}"

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.nx = nx
        self.nu = nu
        self.N = int(N)

        # ---------- QP structure (P, A) ----------

        nx = self.nx
        nu = self.nu
        N = self.N

        # Cost matrix P: block-diag(Q,...,Q,QN,R,...,R)
        QN = Q
        P = sp.block_diag(
            [
                sp.kron(sp.eye(N), Q),  # x_0..x_{N-1}
                QN,                     # x_N
                sp.kron(sp.eye(N), R),  # u_0..u_{N-1}
            ],
            format="csc",
        )
        self.P = P

        # Placeholder q for setup (we update it in solve)
        q0 = np.zeros(P.shape[0])

        # ---------- Dynamics constraints ----------
        # x_{k+1} = A x_k + B u_k
        Ad = sp.csc_matrix(A)
        Bd = sp.csc_matrix(B)

        Ax = sp.kron(sp.eye(N + 1), -sp.eye(nx)) + sp.kron(
            sp.eye(N + 1, k=-1), Ad
        )
        Bu = sp.kron(
            sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]),
            Bd,
        )
        Aeq = sp.hstack([Ax, Bu], format="csc")

        # No inequality constraints yet -> A = Aeq
        A = Aeq
        self.A_mat = A

        # Equality bounds l = u = beq
        leq = np.zeros((N + 1) * nx)
        ueq = np.zeros((N + 1) * nx)
        self.l = leq.copy()
        self.u = ueq.copy()

        # ---------- OSQP problem ----------
        self.prob = osqp.OSQP()
        self.prob.setup(
            P=P,
            q=q0,
            A=A,
            l=self.l,
            u=self.u,
            verbose=verbose,
            warm_start=True,
        )

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
            Currently only the first column is used (constant ref).
        u_ref : np.ndarray, optional
            Not used yet (kept for API compatibility).

        Returns
        -------
        u0 : np.ndarray
            First control action, shape (nu,).
        X_pred : np.ndarray
            Predicted state trajectory, shape (N+1, nx).
        U_pred : np.ndarray
            Predicted input trajectory, shape (N, nu).
        """
        nx = self.nx
        nu = self.nu
        N = self.N

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        assert x0.size == nx, f"x0 must have size {nx}, got {x0.size}"

        # ---- build q from reference ----
        if x_ref is None:
            x_ref_vec = np.zeros(nx)
        else:
            x_ref = np.asarray(x_ref, dtype=float)
            assert x_ref.shape == (nx, N), (
                f"x_ref must have shape ({nx}, {N}), got {x_ref.shape}"
            )
            x_ref_vec = x_ref[:, 0]

        Q = self.Q
        QN = self.Q
        q_x = np.kron(np.ones(N), -Q @ x_ref_vec)
        q_xN = -QN @ x_ref_vec
        q_u = np.zeros(N * nu)
        q = np.hstack([q_x, q_xN, q_u])

        # ---- initial state constraint: l[:nx] = u[:nx] = -x0 ----
        l = self.l.copy()
        u = self.u.copy()
        l[:nx] = -x0
        u[:nx] = -x0

        # ---- update OSQP and solve ----
        self.prob.update(q=q, l=l, u=u)
        res = self.prob.solve()

        if res.info.status_val not in (1,):  # 1 = solved
            raise RuntimeError(f"OSQP did not solve the problem: {res.info.status}")

        z = res.x

        # ---- extract trajectories ----
        x_block_size = (N + 1) * nx
        x_flat = z[:x_block_size]
        u_flat = z[x_block_size:]

        X_pred = x_flat.reshape(N + 1, nx)
        U_pred = u_flat.reshape(N, nu)

        u0 = U_pred[0]

        return u0, X_pred, U_pred
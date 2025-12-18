# nav_mpc/utils/print_solution.py
from __future__ import annotations

import numpy as np


def print_solution(
    i: int,
    x: np.ndarray,
    u0: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    *,
    precision: int = 3,
    max_rows: int | None = None,
    header: bool = True,
) -> None:
    """
    Pretty-print the MPC solution in the terminal.

    Layout: each line corresponds to an index k.
      - For k = 0..N: prints X[k]
      - For k = 0..N-1: prints U[k] on the same row
      - For k = N: prints U as '-' (no input at terminal state)

    Also prints current sim step i, current state x, and applied input u0.

    Args:
        i: simulation step index
        x: current state (nx,)
        u0: applied input (nu,)
        X: predicted state trajectory (N+1, nx)
        U: predicted input trajectory (N, nu)
        precision: numeric formatting precision
        max_rows: optionally truncate printing to first/last rows if long
        header: print a header block
    """
    x = np.asarray(x).reshape(-1)
    u0 = np.asarray(u0).reshape(-1)
    X = np.asarray(X)
    U = np.asarray(U)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N+1, nx), got shape {X.shape}")
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (N, nu), got shape {U.shape}")
    if X.shape[0] != U.shape[0] + 1:
        raise ValueError(f"Expected X rows = U rows + 1, got X{X.shape}, U{U.shape}")

    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]

    # Determine how to show indices if truncating
    def _row_indices() -> list[int]:
        if max_rows is None or (N + 1) <= max_rows:
            return list(range(N + 1))
        # show first half and last half
        first = max_rows // 2
        last = max_rows - first
        return list(range(first)) + [-1] + list(range((N + 1) - last, N + 1))

    rows = _row_indices()

    # Column labels
    x_cols = [f"x{j}" for j in range(nx)]
    u_cols = [f"u{j}" for j in range(nu)]

    # Compute widths
    k_w = max(2, len(str(N)))
    col_w = max(8, precision + 6)  # enough for sign + decimals

    def fmt_num(v: float) -> str:
        return f"{v: {col_w}.{precision}f}"

    def fmt_dash() -> str:
        return " " * (col_w - 1) + "-"

    if header:
        print("\n" + "=" * 80)
        print(f"MPC solution @ sim step i={i}")
        print(f"current x: {np.array2string(x, precision=precision, floatmode='fixed')}")
        print(f"applied u: {np.array2string(u0, precision=precision, floatmode='fixed')}")
        print("-" * 80)

        # Table header
        k_hdr = "k".rjust(k_w)
        hdr = (
            f"{k_hdr} | "
            + "  ".join(name.rjust(col_w) for name in x_cols)
            + " || "
            + "  ".join(name.rjust(col_w) for name in u_cols)
        )
        print(hdr)
        print("-" * len(hdr))

    for k in rows:
        if k == -1:
            # ellipsis row
            dots_x = "  ".join((" " * (col_w - 3) + "...") for _ in range(nx))
            dots_u = "  ".join((" " * (col_w - 3) + "...") for _ in range(nu))
            print(f"{'..'.rjust(k_w)} | {dots_x} || {dots_u}")
            continue

        xk = X[k]
        x_str = "  ".join(fmt_num(float(v)) for v in xk)

        if k < N:
            uk = U[k]
            u_str = "  ".join(fmt_num(float(v)) for v in uk)
        else:
            u_str = "  ".join(fmt_dash() for _ in range(nu))

        print(f"{str(k).rjust(k_w)} | {x_str} || {u_str}")

    if header:
        print("=" * 80 + "\n")

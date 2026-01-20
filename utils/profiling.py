# nav_mpc/utils/profiling.py

from __future__ import annotations

from utils.system_info import print_system_info


def _fmt(ms: float) -> str:
    """Format a time value so it has 3 spaces left and 3 decimals."""
    return f"{ms:7.3f}"   # total width 7, 3 decimals → '   7.464'


def init_timing_stats() -> dict:
    """
    Create and return a fresh timing stats dictionary.
    Times are in milliseconds.
    """
    return {
        # QP evaluation (matrix assembly / update)
        "total_eval":  0.0,
        "min_eval":    float("inf"),
        "max_eval":    0.0,

        # Optimization (OSQP solve)
        "total_opt":   0.0,
        "min_opt":     float("inf"),
        "max_opt":     0.0,

        # Control = QP eval + optimization
        "total_ctrl":  0.0,
        "min_ctrl":    float("inf"),
        "max_ctrl":    0.0,

        # Simulation (system step integration)
        "total_sim":   0.0,
        "min_sim":     float("inf"),
        "max_sim":     0.0,

        # Number of profiled steps
        "n_steps":     0,
    }


def update_timing_stats(
    printing: bool,
    stats: dict,
    start_eval_time: float,
    end_eval_time: float,
    start_opt_time: float,
    end_opt_time: float,
    start_sim_time: float,
    end_sim_time: float,
) -> None:
    """
    Compute per-block times (ms), update running statistics in-place.

    stats keys:
      - total_eval, min_eval, max_eval
      - total_opt,  min_opt,  max_opt
      - total_ctrl, min_ctrl, max_ctrl
      - total_sim,  min_sim,  max_sim
      - n_steps
    """

    # Durations in ms
    eval_time_ms = (end_eval_time - start_eval_time) * 1e3
    opt_time_ms  = (end_opt_time  - start_opt_time)  * 1e3
    sim_time_ms  = (end_sim_time  - start_sim_time)  * 1e3
    ctrl_time_ms = eval_time_ms + opt_time_ms

    # For pretty step index (1-based)
    step_idx = stats["n_steps"] + 1

    if printing:
        print(
            f"Step {step_idx:3d}: "
            f"QP eval time: {_fmt(eval_time_ms)} ms, "
            f"Opt time:     {_fmt(opt_time_ms)} ms, "
            f"Ctrl time:    {_fmt(ctrl_time_ms)} ms, "
            f"Sim time:     {_fmt(sim_time_ms)} ms"
        )

    # Increment number of steps
    stats["n_steps"] += 1

    # QP evaluation
    stats["total_eval"] += eval_time_ms
    stats["min_eval"]    = min(stats["min_eval"], eval_time_ms)
    stats["max_eval"]    = max(stats["max_eval"], eval_time_ms)

    # Optimization
    stats["total_opt"] += opt_time_ms
    stats["min_opt"]    = min(stats["min_opt"], opt_time_ms)
    stats["max_opt"]    = max(stats["max_opt"], opt_time_ms)

    # Control = eval + opt
    stats["total_ctrl"] += ctrl_time_ms
    stats["min_ctrl"]    = min(stats["min_ctrl"], ctrl_time_ms)
    stats["max_ctrl"]    = max(stats["max_ctrl"], ctrl_time_ms)

    # Simulation
    stats["total_sim"] += sim_time_ms
    stats["min_sim"]    = min(stats["min_sim"], sim_time_ms)
    stats["max_sim"]    = max(stats["max_sim"], sim_time_ms)


def print_timing_summary(
    stats: dict,
    dt: float,
    N: int,
    nx: int,
    nu: int,
    nc: int | None = None,
    system_info: bool = True,
) -> None:
    """
    Print a nice summary of timing stats and optionally system info.
    """

    n_steps = max(stats["n_steps"], 1)  # avoid div-by-zero

    avg_eval = stats["total_eval"] / n_steps
    avg_opt  = stats["total_opt"]  / n_steps
    avg_ctrl = stats["total_ctrl"] / n_steps
    avg_sim  = stats["total_sim"]  / n_steps

    print("\n================= Timing statistics over MPC loop =================")
    if nc is None:
        print(f"Problem size: N = {N}, nx = {nx}, nu = {nu}, nc = (unknown / TODO)")
    else:
        print(f"Problem size: N = {N}, nx = {nx}, nu = {nu}, nc = {nc}")

    print(
        "QP update time:     "
        f"avg = {_fmt(avg_eval)} ms, "
        f"min = {_fmt(stats['min_eval'])} ms, "
        f"max = {_fmt(stats['max_eval'])} ms"
    )
    print(
        "QP solution time:   "
        f"avg = {_fmt(avg_opt)} ms, "
        f"min = {_fmt(stats['min_opt'])} ms, "
        f"max = {_fmt(stats['max_opt'])} ms"
    )
    print(
        "Total control time: "
        f"avg = {_fmt(avg_ctrl)} ms, "
        f"min = {_fmt(stats['min_ctrl'])} ms, "
        f"max = {_fmt(stats['max_ctrl'])} ms"
    )
    print(
        "Slack time:         "
        f"avg = {_fmt(dt * 1e3 - avg_ctrl)} ms, "
        f"min = {_fmt(dt * 1e3 - stats['max_ctrl'])} ms, "
        f"max = {_fmt(dt * 1e3 - stats['min_ctrl'])} ms"
    )
    print(
        "Simulation time:    "
        f"avg = {_fmt(avg_sim)} ms, "
        f"min = {_fmt(stats['min_sim'])} ms, "
        f"max = {_fmt(stats['max_sim'])} ms"
    )
    

    if system_info:
        print_system_info()

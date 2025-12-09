# nav_mpc/utils/profiling.py

from utils.system_info import print_system_info


def init_timing_stats() -> dict:
    """
    Create and return a fresh timing stats dictionary.
    """
    return {
        "total_opt":  0.0,
        "min_opt":    float("inf"),
        "max_opt":    0.0,
        "total_sim":  0.0,
        "min_sim":    float("inf"),
        "max_sim":    0.0,
        "total_eval": 0.0,
        "min_eval":   float("inf"),
        "max_eval":   0.0,
        "n_steps":    0,
    }


def update_timing_stats(
    stats: dict,
    start_opt_time: float,
    end_opt_time: float,
    start_sim_time: float,
    end_sim_time: float,
    start_eval_time: float,
    end_eval_time: float,
) -> None:
    """
    Compute per-block times (ms), update running statistics in-place.

    stats is a dict with keys:
      - total_opt, min_opt, max_opt
      - total_sim, min_sim, max_sim
      - total_eval, min_eval, max_eval
      - n_steps
    """

    # Compute durations in ms
    opt_time_ms  = (end_opt_time  - start_opt_time)  * 1e3
    sim_time_ms  = (end_sim_time  - start_sim_time)  * 1e3
    eval_time_ms = (end_eval_time - start_eval_time) * 1e3

    # Increment number of steps
    stats["n_steps"] += 1

    # Optimization
    stats["total_opt"] += opt_time_ms
    stats["min_opt"] = min(stats["min_opt"], opt_time_ms)
    stats["max_opt"] = max(stats["max_opt"], opt_time_ms)

    # Simulation
    stats["total_sim"] += sim_time_ms
    stats["min_sim"] = min(stats["min_sim"], sim_time_ms)
    stats["max_sim"] = max(stats["max_sim"], sim_time_ms)

    # QP evaluation
    stats["total_eval"] += eval_time_ms
    stats["min_eval"] = min(stats["min_eval"], eval_time_ms)
    stats["max_eval"] = max(stats["max_eval"], eval_time_ms)


def print_timing_summary(
    stats: dict,
    N: int,
    nx: int,
    nu: int,
    nc: int | None = None,
    show_system_info: bool = True,
) -> None:
    """
    Print a nice summary of timing stats and optionally system info.
    """

    n_steps = max(stats["n_steps"], 1)  # avoid div-by-zero

    avg_opt  = stats["total_opt"]  / n_steps
    avg_sim  = stats["total_sim"]  / n_steps
    avg_eval = stats["total_eval"] / n_steps

    print("\n================= Timing statistics over MPC loop =================")
    if nc is None:
        print(f"Problem size: N = {N}, nx = {nx}, nu = {nu}, nc = (unknown / TODO)")
    else:
        print(f"Problem size: N = {N}, nx = {nx}, nu = {nu}, nc = {nc}")

    print(
        "Optimization time:   "
        f"avg = {avg_opt:.3f} ms, "
        f"min = {stats['min_opt']:.3f} ms, "
        f"max = {stats['max_opt']:.3f} ms"
    )
    print(
        "Simulation time:     "
        f"avg = {avg_sim:.3f} ms, "
        f"min = {stats['min_sim']:.3f} ms, "
        f"max = {stats['max_sim']:.3f} ms"
    )
    print(
        "QP evaluation time:  "
        f"avg = {avg_eval:.3f} ms, "
        f"min = {stats['min_eval']:.3f} ms, "
        f"max = {stats['max_eval']:.3f} ms"
    )

    if show_system_info:
        print_system_info()

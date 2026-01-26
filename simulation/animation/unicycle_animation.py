# nav_mpc/simulation/animation/unicycle_animation.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm

from core.models.unicycle_kinematic_model import UnicycleKinematicModel


def _resolve_results_dir(save_path: str | Path | None) -> Tuple[Path, Path]:
    """
    Returns (results_dir, mp4_path).

    If save_path is None:
        <project_root>/results/unicycle_animation.mp4
    If save_path is a directory:
        <save_path>/unicycle_animation.mp4
    If save_path is a file:
        that exact file path (and results_dir = parent)
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "unicycle_animation.mp4"

    save_path = Path(save_path)
    if save_path.is_dir():
        results_dir = save_path
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "unicycle_animation.mp4"

    results_dir = save_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, save_path


def _infer_wheel_speed_bounds(system: UnicycleKinematicModel, constraints: object | None) -> Tuple[float, float]:
    """
    Infer wheel speed bounds ±omega_max from constraints (preferred) or fallback.
    """
    if constraints is not None and hasattr(constraints, "omega_max"):
        wmax = float(getattr(constraints, "omega_max"))
        return -wmax, +wmax
    return -15.0, 15.0


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _square_body_vertices(px: float, py: float, phi: float, half_side: float) -> np.ndarray:
    """
    Square centered at (px,py), side = 2*half_side, rotated by phi.
    Returns (4,2) vertices in CCW order.
    """
    pts = np.array(
        [
            [-half_side, -half_side],
            [+half_side, -half_side],
            [+half_side, +half_side],
            [-half_side, +half_side],
        ],
        dtype=float,
    )

    c = float(np.cos(phi))
    s = float(np.sin(phi))
    Rm = np.array([[c, -s], [s, c]], dtype=float)

    return (pts @ Rm.T) + np.array([px, py], dtype=float)


def _extract_pred_list(kwargs: dict) -> list[np.ndarray] | None:
    for key in ("X_pred_traj", "X_pred_list", "X_preds", "X_horizon", "X_hist"):
        if key in kwargs and kwargs[key] is not None:
            pred = kwargs[key]
            if isinstance(pred, list):
                return pred
            arr = np.asarray(pred, dtype=float)
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def _extract_ref_list(kwargs: dict) -> list[np.ndarray] | None:
    for key in ("X_ref_traj", "X_ref_list", "Xref_traj", "Xrefs"):
        if key in kwargs and kwargs[key] is not None:
            ref = kwargs[key]
            if isinstance(ref, list):
                return ref
            arr = np.asarray(ref, dtype=float)
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def _extract_scan_list(kwargs: dict) -> list | None:
    for key in ("lidar_scans", "scans", "scan_list"):
        if key in kwargs and kwargs[key] is not None:
            scans = kwargs[key]
            if isinstance(scans, list):
                return scans
            arr = np.asarray(scans)
            if arr.ndim == 2:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def _extract_global_path(kwargs: dict) -> np.ndarray | None:
    for key in ("global_path", "path_waypoints"):
        if key in kwargs and kwargs[key] is not None:
            gp = np.asarray(kwargs[key], dtype=float)
            if gp.ndim == 2 and gp.shape[0] >= 1 and gp.shape[1] >= 2:
                return gp[:, :2].copy()
    return None


def _scan_to_points_robot(scan) -> np.ndarray:
    """
    Convert a LaserScanLike (or raw ranges array) to ROBOT-frame points (M,2).
    Robot frame: +x forward, +y left.
    """
    if hasattr(scan, "ranges"):
        ranges = np.asarray(scan.ranges, dtype=float).reshape(-1)
        angle_min = float(scan.angle_min)
        angle_inc = float(scan.angle_increment)
        angles = angle_min + np.arange(ranges.size) * angle_inc
    else:
        ranges = np.asarray(scan, dtype=float).reshape(-1)
        angles = np.linspace(-np.pi, np.pi, ranges.size, endpoint=True)

    mask = np.isfinite(ranges)
    if not np.any(mask):
        return np.empty((0, 2), dtype=float)

    r = ranges[mask]
    a = angles[mask]

    xr = r * np.cos(a)
    yr = r * np.sin(a)
    return np.column_stack([xr, yr])


def _points_robot_to_world(pts_r: np.ndarray, px: float, py: float, yaw: float) -> np.ndarray:
    """Rigid transform from robot frame to world frame."""
    if pts_r.size == 0:
        return pts_r.reshape(0, 2)

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    xw = px + c * pts_r[:, 0] - s * pts_r[:, 1]
    yw = py + s * pts_r[:, 0] + c * pts_r[:, 1]
    return np.column_stack([xw, yw])


def _maybe_load_occ_background(kwargs):
    """Returns (occ_img, extent) or (None, None)."""
    try:
        from simulation.environment.occupancy_map import OccupancyMap2D, OccupancyMapConfig
    except Exception:
        return None, None

    occ_map = kwargs.get("occ_map", None)
    if occ_map is not None:
        occ = np.flipud(occ_map.occ).astype(float)
        extent = [occ_map.xmin, occ_map.xmax, occ_map.ymin, occ_map.ymax]
        return occ, extent

    occ_cfg = kwargs.get("occ_cfg", None)
    if occ_cfg is not None:
        occ_map = OccupancyMap2D.from_png(occ_cfg)
        occ = np.flipud(occ_map.occ).astype(float)
        extent = [occ_map.xmin, occ_map.xmax, occ_map.ymin, occ_map.ymax]
        return occ, extent

    return None, None


def _extract_col_bounds_list(kwargs: dict) -> list[np.ndarray | None] | None:
    for key in ("col_bounds_traj", "col_B_traj", "collision_bounds_traj"):
        if key in kwargs and kwargs[key] is not None:
            lst = kwargs[key]
            if isinstance(lst, list):
                return lst
    return None


def _extract_col_Axy_list(kwargs: dict) -> list[np.ndarray | None] | None:
    for key in ("col_Axy_traj", "col_Axy_list", "collision_Axy_traj"):
        if key in kwargs and kwargs[key] is not None:
            lst = kwargs[key]
            if isinstance(lst, list):
                return lst
    return None


def _extract_collision_normals(kwargs: dict) -> np.ndarray | None:
    """
    Backwards-compatible fallback: if you pass a collision config with fixed normals.
    """
    col = kwargs.get("collision", None)
    if col is not None and hasattr(col, "normals"):
        try:
            N = col.normals()
        except TypeError:
            N = col.normals
        N = np.asarray(N, dtype=float)
        if N.ndim == 2 and N.shape[1] == 2:
            return N
    if "collision_normals" in kwargs and kwargs["collision_normals"] is not None:
        N = np.asarray(kwargs["collision_normals"], dtype=float)
        if N.ndim == 2 and N.shape[1] == 2:
            return N
    return None


def _halfspace_intersection_polygon(normals: np.ndarray, b: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """
    Legacy: Compute vertices of intersection {p | n_i^T p <= b_i} clipped by bbox.
    """
    xmin, xmax, ymin, ymax = map(float, bbox)

    normals = np.asarray(normals, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if normals.ndim != 2 or normals.shape[1] != 2 or b.size != normals.shape[0]:
        return np.empty((0, 2), dtype=float)

    Nb = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    bb = np.array([xmax, -xmin, ymax, -ymin], dtype=float)

    Nall = np.vstack([normals, Nb])
    ball = np.concatenate([b, bb])

    M = Nall.shape[0]
    pts = []

    for i in range(M):
        ni = Nall[i]
        bi = ball[i]
        for j in range(i + 1, M):
            nj = Nall[j]
            bj = ball[j]

            A = np.array([ni, nj], dtype=float)
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-10:
                continue

            p = np.linalg.solve(A, np.array([bi, bj], dtype=float))

            if np.all(Nall @ p <= ball + 1e-9):
                pts.append(p)

    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    P = np.vstack(pts)

    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    order = np.argsort(ang)
    P = P[order]

    keep = [0]
    for k in range(1, P.shape[0]):
        if np.linalg.norm(P[k] - P[keep[-1]]) > 1e-6:
            keep.append(k)
    P = P[keep]

    if P.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    return P


def _halfspace_intersection_polygon_Axb(A: np.ndarray, b: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """
    Compute vertices of intersection {p | A p <= b} clipped by bbox.

    A: (M,2), b: (M,)
    bbox = (xmin, xmax, ymin, ymax)
    Returns vertices (K,2) in CCW order, or empty (0,2).
    """
    xmin, xmax, ymin, ymax = map(float, bbox)

    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.shape[0] != b.size:
        return np.empty((0, 2), dtype=float)

    # Filter inactive / degenerate rows (e.g., [0,0])
    row_norm = np.linalg.norm(A, axis=1)
    mask = row_norm > 1e-12
    A = A[mask]
    b = b[mask]

    if A.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    # Add bbox halfspaces
    Ab = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    bb = np.array([xmax, -xmin, ymax, -ymin], dtype=float)

    Aall = np.vstack([A, Ab])
    ball = np.concatenate([b, bb])

    M = Aall.shape[0]
    pts = []

    for i in range(M):
        ai = Aall[i]
        bi = ball[i]
        for j in range(i + 1, M):
            aj = Aall[j]
            bj = ball[j]

            Mat = np.array([ai, aj], dtype=float)
            det = Mat[0, 0] * Mat[1, 1] - Mat[0, 1] * Mat[1, 0]
            if abs(det) < 1e-12:
                continue

            p = np.linalg.solve(Mat, np.array([bi, bj], dtype=float))

            if np.all(Aall @ p <= ball + 1e-9):
                pts.append(p)

    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    P = np.vstack(pts)

    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    order = np.argsort(ang)
    P = P[order]

    keep = [0]
    for k in range(1, P.shape[0]):
        if np.linalg.norm(P[k] - P[keep[-1]]) > 1e-6:
            keep.append(k)
    P = P[keep]

    if P.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    return P


def animate_unicycle(
    system: UnicycleKinematicModel,
    constraints: object | None,
    dt: float,
    x_traj: list[np.ndarray] | np.ndarray,
    u_traj: list[np.ndarray] | np.ndarray,
    x_goal: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    save_gif: bool = False,
    **kwargs,
):
    """
    Call signature kept identical to main():

        animation(system=system, constraints=constraints, dt=dt,
                  x_traj=x_traj, u_traj=u_traj, x_goal=x_goal,
                  X_pred_traj=X_pred_traj, X_ref_traj=X_ref_traj,
                  lidar_scans=scans, occ_map=occ_map, global_path=global_path,
                  collision=collision,
                  col_bounds_traj=col_bounds_traj,
                  col_Axy_traj=col_Axy_traj,
                  show=False, save_gif=True)

    Model:
      x = [px, py, phi, v, r]    (T,5)
      u = [alpha_l, alpha_r]     (T-1,2)

    Wheel speeds derived from state (v,r):
      omega_l = (v - L*r)/R
      omega_r = (v + L*r)/R
    """
    X_pred_list = _extract_pred_list(kwargs)
    X_ref_list = _extract_ref_list(kwargs)
    scan_list = _extract_scan_list(kwargs)
    occ_img, occ_extent = _maybe_load_occ_background(kwargs)
    global_path = _extract_global_path(kwargs)

    col_bounds_list = _extract_col_bounds_list(kwargs)  # list of (N,M) or None
    col_Axy_list = _extract_col_Axy_list(kwargs)        # list of (N,M,2) or None

    # Backwards-compatible fallback (fixed normals corridor)
    collision_normals = _extract_collision_normals(kwargs)

    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 110
    GIF_DPI = 80

    N_GHOSTS = 5
    GHOST_ALPHA_MIN = 0.2
    PLAN_LINE_WIDTH = 2.0
    LIDAR_POINT_SIZE = 6

    GP_LINE_WIDTH = 1.5
    GP_MARKER_SIZE = 2

    # ---- Trajectory arrays ----
    x_arr = np.vstack(x_traj) if isinstance(x_traj, list) else np.asarray(x_traj, dtype=float)
    u_arr = np.vstack(u_traj) if isinstance(u_traj, list) else np.asarray(u_traj, dtype=float)

    if x_arr.ndim != 2 or x_arr.shape[1] != 5:
        raise ValueError(f"Expected x_traj shape (T,5), got {x_arr.shape}")
    if u_arr.ndim != 2 or u_arr.shape[1] != 2:
        raise ValueError(f"Expected u_traj shape (T-1,2), got {u_arr.shape}")

    T = x_arr.shape[0]
    if u_arr.shape[0] != T - 1:
        raise ValueError(f"Expected u_traj length T-1={T-1}, got {u_arr.shape[0]}")

    t = dt * np.arange(T, dtype=float)

    px = x_arr[:, 0].astype(float)
    py = x_arr[:, 1].astype(float)
    phi = _wrap_to_pi(x_arr[:, 2].astype(float))
    v = x_arr[:, 3].astype(float)
    r = x_arr[:, 4].astype(float)

    alpha_l = u_arr[:, 0].astype(float)
    alpha_r = u_arr[:, 1].astype(float)

    Rw = float(getattr(system, "R", 0.040))
    Lh = float(getattr(system, "L", 0.062))

    omega_l = (v - Lh * r) / Rw
    omega_r = (v + Lh * r) / Rw

    goal_xy = None
    if x_goal is not None:
        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        if x_goal.size >= 2:
            goal_xy = x_goal[:2].copy()

    half_side = float(getattr(system, "body_size", 0.08))
    if half_side <= 0.0:
        half_side = 0.08

    wmin, wmax = _infer_wheel_speed_bounds(system, constraints)
    wabs = max(abs(wmin), abs(wmax), 1e-6)

    sim_fps = 1.0 / float(dt)
    frame_stride = max(1, int(round(sim_fps / TARGET_ANIM_FPS)))

    frame_indices = np.arange(0, T, frame_stride)
    if frame_indices[-1] != T - 1:
        frame_indices = np.append(frame_indices, T - 1)

    t_anim = t[frame_indices]
    T_anim = frame_indices.size

    fig, (ax_xy, ax_w) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"wspace": 0.35})
    fig.suptitle("Unicycle simulation")

    # -------- XY axis --------
    ax_xy.set_aspect("equal", "box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True)

    margin = 3.0 * half_side
    xmin, xmax = float(px.min() - margin), float(px.max() + margin)
    ymin, ymax = float(py.min() - margin), float(py.max() + margin)

    if goal_xy is not None:
        xmin = min(xmin, float(goal_xy[0]) - margin)
        xmax = max(xmax, float(goal_xy[0]) + margin)
        ymin = min(ymin, float(goal_xy[1]) - margin)
        ymax = max(ymax, float(goal_xy[1]) + margin)

    if global_path is not None:
        xmin = min(xmin, float(np.min(global_path[:, 0])) - margin)
        xmax = max(xmax, float(np.max(global_path[:, 0])) + margin)
        ymin = min(ymin, float(np.min(global_path[:, 1])) - margin)
        ymax = max(ymax, float(np.max(global_path[:, 1])) + margin)

    if occ_extent is not None:
        xmin = min(xmin, float(occ_extent[0]))
        xmax = max(xmax, float(occ_extent[1]))
        ymin = min(ymin, float(occ_extent[2]))
        ymax = max(ymax, float(occ_extent[3]))

    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)

    if occ_img is not None and occ_extent is not None:
        cmap = ListedColormap(["#ffefb0", "#72498D"])  # free, occupied
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        ax_xy.imshow(
            occ_img,
            origin="lower",
            extent=occ_extent,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            alpha=0.35,
            zorder=0,
        )

    if global_path is not None and global_path.shape[0] >= 1:
        ax_xy.plot(
            global_path[:, 0],
            global_path[:, 1],
            linewidth=GP_LINE_WIDTH,
            linestyle="--",
            color="#ff7f0e",
            alpha=0.25,
        )
        ax_xy.scatter(
            global_path[:, 0],
            global_path[:, 1],
            s=GP_MARKER_SIZE,
            marker=".",
            color="#ff7f0e",
            alpha=0.0,
        )

    ax_xy.plot(px[0], py[0], marker=".", markersize=3, linestyle="None")
    if goal_xy is not None:
        ax_xy.plot(goal_xy[0], goal_xy[1], marker="*", color="#ff7f0e", markersize=8, linestyle="None")

    (trail_line,) = ax_xy.plot([], [], linestyle="--", color="#008000", linewidth=1.5)
    (plan_line,) = ax_xy.plot([], [], linewidth=PLAN_LINE_WIDTH, color="#0058ca", linestyle="dotted")
    lidar_scatter = ax_xy.scatter([], [], s=LIDAR_POINT_SIZE, marker=".", color="#b11010", alpha=0.8)
    (ref_line,) = ax_xy.plot([], [], linewidth=2.0, linestyle="-", color="#72498D", alpha=0.1)
    ref_scatter = ax_xy.scatter([], [], s=25, marker="o", color="#72498D", alpha=0.1)

    # ---- Ghosts and collision polytopes are synced by the SAME N_GHOSTS ----
    ghost_alphas = np.linspace(1.0, GHOST_ALPHA_MIN, N_GHOSTS + 1)[1:]

    # Collision corridor polygons: one per ghost (same count, same alpha fade)
    col_polys: list[Polygon] = []
    have_corridors = (col_Axy_list is not None and col_bounds_list is not None) or (
        collision_normals is not None and col_bounds_list is not None
    )
    if have_corridors:
        for a in ghost_alphas:
            poly = Polygon(
                np.zeros((0, 2), dtype=float),
                closed=True,
                facecolor="none",
                edgecolor="#00FF00",
                linewidth=2.0,
                alpha=float(a) * 0.9,
                zorder=4,
            )
            ax_xy.add_patch(poly)
            col_polys.append(poly)

    body_poly = Polygon(
        _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
        closed=True,
        facecolor="#1f77b4",
        edgecolor="#1f77b4",
        linewidth=0.1,
        alpha=1.0,
        zorder=6,
    )
    ax_xy.add_patch(body_poly)

    ghost_polys: list[Polygon] = []
    if X_pred_list is not None:
        for a in ghost_alphas:
            gp = Polygon(
                _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
                closed=True,
                alpha=float(a),
            )
            ax_xy.add_patch(gp)
            ghost_polys.append(gp)

    (heading_line,) = ax_xy.plot([], [], linewidth=2)
    time_text = ax_xy.text(0.02, 0.95, "", transform=ax_xy.transAxes)

    # -------- wheel speed axis --------
    ax_w.set_title("Wheel speeds (derived)")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.grid(True)
    ax_w.set_xlim(float(t[0]), float(t[-1]))
    ax_w.set_ylim(-1.1 * wabs, 1.1 * wabs)
    ax_w.axhline(0.0, color="k", linewidth=0.8, alpha=0.8)
    ax_w.axhline(float(wmin), linestyle="--", linewidth=1.5, color="#b11010", alpha=0.3)
    ax_w.axhline(float(wmax), linestyle="--", linewidth=1.5, color="#b11010", alpha=0.3)

    (w_l_line,) = ax_w.plot([], [], linewidth=2)
    (w_r_line,) = ax_w.plot([], [], linewidth=2)

    def _set_body(poly: Polygon, px_k: float, py_k: float, phi_k: float) -> None:
        poly.set_xy(_square_body_vertices(px_k, py_k, phi_k, half_side))

    def _set_heading(px_k: float, py_k: float, phi_k: float) -> None:
        hlen = 1.2 * half_side
        xh = px_k + hlen * float(np.cos(phi_k))
        yh = py_k + hlen * float(np.sin(phi_k))
        heading_line.set_data([px_k, xh], [py_k, yh])

    def _compute_ghost_indices(pred: np.ndarray) -> np.ndarray:
        """
        pred is (Nh+1, nx). Return indices into pred stages, length up to N_GHOSTS.
        We use the same sampling as the ghosts and corridors.
        """
        Nh = pred.shape[0] - 1
        if Nh <= 0:
            return np.array([], dtype=int)
        idxs = np.linspace(0, Nh, N_GHOSTS + 1).round().astype(int)
        return idxs[1:]  # drop 0, keep future stages

    def _set_plan_and_ghosts(step_k: int) -> np.ndarray:
        """
        Updates plan line and ghost robot bodies.
        Returns ghost_idxs so we can sync collision polytopes to the SAME stages.
        """
        if X_pred_list is None:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_visible(False)
            return np.array([], dtype=int)

        last_cycle = min(max(step_k, 0), len(X_pred_list) - 1)
        pred = np.asarray(X_pred_list[last_cycle], dtype=float)

        if pred.ndim != 2 or pred.shape[1] < 3:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_visible(False)
            return np.array([], dtype=int)

        plan_line.set_data(pred[:, 0], pred[:, 1])

        ghost_idxs = _compute_ghost_indices(pred)
        if ghost_idxs.size == 0:
            for gp in ghost_polys:
                gp.set_visible(False)
            return ghost_idxs

        for gp, ii in zip(ghost_polys, ghost_idxs, strict=False):
            _set_body(
                gp,
                float(pred[ii, 0]),
                float(pred[ii, 1]),
                float(_wrap_to_pi(np.array([pred[ii, 2]])).item()),
            )
            gp.set_visible(True)

        for gp in ghost_polys[len(ghost_idxs):]:
            gp.set_visible(False)

        return ghost_idxs

    def _set_reference(step_k: int) -> None:
        if X_ref_list is None:
            ref_line.set_data([], [])
            ref_scatter.set_offsets(np.empty((0, 2)))
            return

        last_cycle = min(max(step_k, 0), len(X_ref_list) - 1)
        ref = np.asarray(X_ref_list[last_cycle], dtype=float)

        if ref.ndim != 2 or ref.shape[1] < 2:
            ref_line.set_data([], [])
            ref_scatter.set_offsets(np.empty((0, 2)))
            return

        ref_line.set_data(ref[:, 0], ref[:, 1])
        ref_scatter.set_offsets(ref[:, :2])

    def _set_lidar(step_k: int) -> None:
        if scan_list is None or step_k < 0 or step_k >= len(scan_list):
            lidar_scatter.set_offsets(np.empty((0, 2)))
            return

        pts_r = _scan_to_points_robot(scan_list[step_k])
        pts_w = _points_robot_to_world(pts_r, float(px[step_k]), float(py[step_k]), float(phi[step_k]))
        lidar_scatter.set_offsets(pts_w)

    def _set_collision_polys(step_k: int, ghost_idxs: np.ndarray) -> None:
        """
        Updates collision corridor polytopes synced to the SAME stages as ghosts.

        Preferred:
          uses per-stage halfspaces A_xy[ii,:,:] p <= b[ii,:]

        Fallback:
          uses fixed normals N p <= B[ii,:]
        """
        if len(col_polys) == 0:
            return

        if (col_bounds_list is None) or (step_k < 0) or (step_k >= len(col_bounds_list)):
            for cp in col_polys:
                cp.set_visible(False)
            return

        Bk = col_bounds_list[step_k]
        if Bk is None:
            for cp in col_polys:
                cp.set_visible(False)
            return
        Bk = np.asarray(Bk, dtype=float)

        if ghost_idxs.size == 0:
            for cp in col_polys:
                cp.set_visible(False)
            return

        xmin, xmax = ax_xy.get_xlim()
        ymin, ymax = ax_xy.get_ylim()
        bbox = (xmin, xmax, ymin, ymax)

        # Preferred: per-stage Axy constraints
        if col_Axy_list is not None and step_k < len(col_Axy_list):
            Axyk = col_Axy_list[step_k]
            if Axyk is None:
                for cp in col_polys:
                    cp.set_visible(False)
                return

            Axyk = np.asarray(Axyk, dtype=float)
            if Axyk.ndim != 3 or Axyk.shape[0] != Bk.shape[0] or Axyk.shape[1] != Bk.shape[1] or Axyk.shape[2] != 2:
                for cp in col_polys:
                    cp.set_visible(False)
                return

            for cp, ii in zip(col_polys, ghost_idxs, strict=False):
                if ii < 0 or ii >= Bk.shape[0]:
                    cp.set_visible(False)
                    continue

                Aii = Axyk[ii, :, :]  # (M,2)
                bii = Bk[ii, :]       # (M,)
                verts = _halfspace_intersection_polygon_Axb(Aii, bii, bbox=bbox)

                if verts.shape[0] < 3:
                    cp.set_visible(False)
                else:
                    cp.set_xy(verts)
                    cp.set_visible(True)

            for cp in col_polys[len(ghost_idxs):]:
                cp.set_visible(False)
            return

        # Fallback: fixed normals corridor
        if collision_normals is None or Bk.ndim != 2 or Bk.shape[1] != collision_normals.shape[0]:
            for cp in col_polys:
                cp.set_visible(False)
            return

        for cp, ii in zip(col_polys, ghost_idxs, strict=False):
            if ii < 0 or ii >= Bk.shape[0]:
                cp.set_visible(False)
                continue

            verts = _halfspace_intersection_polygon(collision_normals, Bk[ii, :], bbox=bbox)
            if verts.shape[0] < 3:
                cp.set_visible(False)
            else:
                cp.set_xy(verts)
                cp.set_visible(True)

        for cp in col_polys[len(ghost_idxs):]:
            cp.set_visible(False)

    def init():
        trail_line.set_data([], [])
        plan_line.set_data([], [])
        lidar_scatter.set_offsets(np.empty((0, 2)))
        ref_line.set_data([], [])
        ref_scatter.set_offsets(np.empty((0, 2)))

        _set_body(body_poly, float(px[0]), float(py[0]), float(phi[0]))
        _set_heading(float(px[0]), float(py[0]), float(phi[0]))
        time_text.set_text("")

        w_l_line.set_data([], [])
        w_r_line.set_data([], [])

        for gp in ghost_polys:
            gp.set_visible(False)
        for cp in col_polys:
            cp.set_visible(False)

        artists = [
            trail_line,
            plan_line,
            ref_line,
            ref_scatter,
            lidar_scatter,
            body_poly,
            heading_line,
            time_text,
            w_l_line,
            w_r_line,
        ]
        artists.extend(ghost_polys)
        artists.extend(col_polys)
        return tuple(artists)

    def update(frame_idx: int):
        k = int(frame_indices[frame_idx])

        trail_line.set_data(px[: k + 1], py[: k + 1])
        _set_body(body_poly, float(px[k]), float(py[k]), float(phi[k]))
        _set_heading(float(px[k]), float(py[k]), float(phi[k]))

        ghost_idxs = _set_plan_and_ghosts(step_k=k)
        _set_lidar(step_k=k)
        _set_reference(step_k=k)
        _set_collision_polys(step_k=k, ghost_idxs=ghost_idxs)

        w_l_line.set_data(t[: k + 1], omega_l[: k + 1])
        w_r_line.set_data(t[: k + 1], omega_r[: k + 1])

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")

        artists = [
            trail_line,
            plan_line,
            ref_line,
            ref_scatter,
            lidar_scatter,
            body_poly,
            heading_line,
            time_text,
            w_l_line,
            w_r_line,
        ]
        artists.extend(ghost_polys)
        artists.extend(col_polys)
        return tuple(artists)

    ani = FuncAnimation(
        fig,
        update,
        frames=T_anim,
        init_func=init,
        blit=True,
        interval=1000.0 / TARGET_ANIM_FPS,
    )

    _, mp4_path = _resolve_results_dir(save_path)
    video_fps = int(TARGET_ANIM_FPS)

    ani.save(mp4_path, fps=video_fps, dpi=VIDEO_DPI)
    print(f"[animator] Saved animation to {mp4_path}")

    if save_gif:
        gif_path = mp4_path.with_suffix(".gif")
        writer = PillowWriter(fps=video_fps)
        ani.save(gif_path, writer=writer, dpi=GIF_DPI)
        print(f"[animator] Saved GIF animation to {gif_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani

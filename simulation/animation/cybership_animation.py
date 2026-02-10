from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm


def _resolve_results_dir(save_path: str | Path | None) -> Tuple[Path, Path]:
    """
    Returns (results_dir, mp4_path).

    If save_path is None:
        <project_root>/results/cybership_animation.mp4
    If save_path is a directory:
        <save_path>/cybership_animation.mp4
    If save_path is a file:
        that exact file path (and results_dir = parent)
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "cybership_animation.mp4"

    save_path = Path(save_path)
    if save_path.is_dir():
        results_dir = save_path
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "cybership_animation.mp4"

    results_dir = save_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, save_path


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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

    # de-dup
    keep = [0]
    for k in range(1, P.shape[0]):
        if np.linalg.norm(P[k] - P[keep[-1]]) > 1e-6:
            keep.append(k)
    P = P[keep]

    if P.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    return P


def _ship_body_vertices(px: float, py: float, psi: float, L: float, B: float) -> np.ndarray:
    """
    Simple ship silhouette (polygon) in body frame, then rotated to world.

    L: length [m] (approx)
    B: beam  [m] (approx)
    """
    # A slightly pointy bow + flat stern. CCW.
    # Body frame: +x forward, +y left.
    pts = np.array(
        [
            [+0.5 * L,  0.0],          # bow
            [+0.2 * L, +0.5 * B],      # port-fore
            [-0.5 * L, +0.5 * B],      # port-stern
            [-0.5 * L, -0.5 * B],      # starboard-stern
            [+0.2 * L, -0.5 * B],      # starboard-fore
        ],
        dtype=float,
    )

    c = float(np.cos(psi))
    s = float(np.sin(psi))
    Rm = np.array([[c, -s], [s, c]], dtype=float)

    return (pts @ Rm.T) + np.array([px, py], dtype=float)


def animate_cybership(
    system,
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
    Same "main()" compatible signature style as unicycle:

        animation(system=system, constraints=constraints, dt=dt,
                  x_traj=x_traj, u_traj=u_traj, x_goal=x_goal,
                  X_pred_traj=..., X_ref_traj=...,
                  lidar_scans=..., occ_map=..., global_path=...,
                  col_bounds_traj=..., col_Axy_traj=...,
                  show=False, save_gif=True)

    Model:
      x = [px, py, psi, ux, uy, r]      (T,6)
      u = [Tl, Tr, Tb, az_l, az_r]      (T-1,5)
    """
    X_pred_list = _extract_pred_list(kwargs)
    X_ref_list = _extract_ref_list(kwargs)
    scan_list = _extract_scan_list(kwargs)
    occ_img, occ_extent = _maybe_load_occ_background(kwargs)
    global_path = _extract_global_path(kwargs)

    col_bounds_list = _extract_col_bounds_list(kwargs)
    col_Axy_list = _extract_col_Axy_list(kwargs)

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

    if x_arr.ndim != 2 or x_arr.shape[1] < 6:
        raise ValueError(f"Expected x_traj shape (T,>=6), got {x_arr.shape}")
    if u_arr.ndim != 2 or u_arr.shape[1] < 5:
        raise ValueError(f"Expected u_traj shape (T-1,>=5), got {u_arr.shape}")

    T = x_arr.shape[0]
    if u_arr.shape[0] != T - 1:
        raise ValueError(f"Expected u_traj length T-1={T-1}, got {u_arr.shape[0]}")

    t = dt * np.arange(T, dtype=float)

    px = x_arr[:, 0].astype(float)
    py = x_arr[:, 1].astype(float)
    psi = _wrap_to_pi(x_arr[:, 2].astype(float))
    ux = x_arr[:, 3].astype(float)
    uy = x_arr[:, 4].astype(float)
    r = x_arr[:, 5].astype(float)

    Tl = u_arr[:, 0].astype(float)
    Tr = u_arr[:, 1].astype(float)
    Tb = u_arr[:, 2].astype(float)
    az_l = _wrap_to_pi(u_arr[:, 3].astype(float))
    az_r = _wrap_to_pi(u_arr[:, 4].astype(float))

    # pad inputs to length T (for plotting step k)
    Tl_plot = np.r_[Tl, Tl[-1]]
    Tr_plot = np.r_[Tr, Tr[-1]]
    Tb_plot = np.r_[Tb, Tb[-1]]
    az_l_plot = np.r_[az_l, az_l[-1]]
    az_r_plot = np.r_[az_r, az_r[-1]]

    goal_xy = None
    if x_goal is not None:
        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        if x_goal.size >= 2:
            goal_xy = x_goal[:2].copy()

    # Try to infer ship size from model, fallback to something visible
    L = float(getattr(system, "l", 1.255))
    B = float(getattr(system, "b", 0.29))
    if not np.isfinite(L) or L <= 0:
        L = 1.2
    if not np.isfinite(B) or B <= 0:
        B = 0.3

    margin = 0.8 * L

    sim_fps = 1.0 / float(dt)
    frame_stride = max(1, int(round(sim_fps / TARGET_ANIM_FPS)))

    frame_indices = np.arange(0, T, frame_stride)
    if frame_indices[-1] != T - 1:
        frame_indices = np.append(frame_indices, T - 1)

    t_anim = t[frame_indices]
    T_anim = frame_indices.size

    fig, (ax_xy, ax_u) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"wspace": 0.35})
    fig.suptitle("Cybership simulation")

    # -------- XY axis --------
    ax_xy.set_aspect("equal", "box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True)

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
    (ref_line,) = ax_xy.plot([], [], linewidth=2.0, linestyle="-", color="#72498D", alpha=0.15)
    ref_scatter = ax_xy.scatter([], [], s=20, marker="o", color="#72498D", alpha=0.15)

    ghost_alphas = np.linspace(1.0, GHOST_ALPHA_MIN, N_GHOSTS + 1)[1:]

    ship_poly = Polygon(
        _ship_body_vertices(float(px[0]), float(py[0]), float(psi[0]), L=L, B=B),
        closed=True,
        facecolor="#1f77b4",
        edgecolor="#1f77b4",
        linewidth=0.1,
        alpha=1.0,
        zorder=6,
    )
    ax_xy.add_patch(ship_poly)

    ghost_polys: list[Polygon] = []
    if X_pred_list is not None:
        for a in ghost_alphas:
            gp = Polygon(
                _ship_body_vertices(float(px[0]), float(py[0]), float(psi[0]), L=L, B=B),
                closed=True,
                alpha=float(a),
            )
            ax_xy.add_patch(gp)
            ghost_polys.append(gp)

    # Collision corridor polygons: one per ghost, if available
    col_polys: list[Polygon] = []
    have_corridors = (col_Axy_list is not None and col_bounds_list is not None)
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

    (heading_line,) = ax_xy.plot([], [], linewidth=2)
    time_text = ax_xy.text(0.02, 0.95, "", transform=ax_xy.transAxes)

    # -------- Inputs axis --------
    ax_u.set_title("Inputs")
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("thrust [N] / azimuth [rad]")
    ax_u.grid(True)
    ax_u.set_xlim(float(t[0]), float(t[-1]))

    # Autoscale y based on thrust and angle range
    thrust_abs = float(np.max(np.abs(np.r_[Tl_plot, Tr_plot, Tb_plot, 1e-6])))
    y_lo = min(-1.1 * thrust_abs, -np.pi - 0.2)
    y_hi = max(+1.1 * thrust_abs, +np.pi + 0.2)
    ax_u.set_ylim(y_lo, y_hi)
    ax_u.axhline(0.0, color="k", linewidth=0.8, alpha=0.8)
    ax_u.axhline(-np.pi, linestyle="--", linewidth=1.0, alpha=0.25)
    ax_u.axhline(+np.pi, linestyle="--", linewidth=1.0, alpha=0.25)

    (tl_line,) = ax_u.plot([], [], linewidth=2, label="thrust_left")
    (tr_line,) = ax_u.plot([], [], linewidth=2, label="thrust_right")
    (tb_line,) = ax_u.plot([], [], linewidth=2, label="thrust_bow")
    (azl_line,) = ax_u.plot([], [], linewidth=2, label="az_left")
    (azr_line,) = ax_u.plot([], [], linewidth=2, label="az_right")
    ax_u.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _set_ship(poly: Polygon, px_k: float, py_k: float, psi_k: float) -> None:
        poly.set_xy(_ship_body_vertices(px_k, py_k, psi_k, L=L, B=B))

    def _set_heading(px_k: float, py_k: float, psi_k: float) -> None:
        hlen = 0.7 * L
        xh = px_k + hlen * float(np.cos(psi_k))
        yh = py_k + hlen * float(np.sin(psi_k))
        heading_line.set_data([px_k, xh], [py_k, yh])

    def _compute_ghost_indices(pred: np.ndarray) -> np.ndarray:
        Nh = pred.shape[0] - 1
        if Nh <= 0:
            return np.array([], dtype=int)
        idxs = np.linspace(0, Nh, N_GHOSTS + 1).round().astype(int)
        return idxs[1:]

    def _set_plan_and_ghosts(step_k: int) -> np.ndarray:
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
            _set_ship(
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
        pts_w = _points_robot_to_world(pts_r, float(px[step_k]), float(py[step_k]), float(psi[step_k]))
        lidar_scatter.set_offsets(pts_w)

    def _set_collision_polys(step_k: int, ghost_idxs: np.ndarray) -> None:
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

        if col_Axy_list is None or step_k >= len(col_Axy_list):
            for cp in col_polys:
                cp.set_visible(False)
            return

        Axyk = col_Axy_list[step_k]
        if Axyk is None:
            for cp in col_polys:
                cp.set_visible(False)
            return
        Axyk = np.asarray(Axyk, dtype=float)

        xmin, xmax = ax_xy.get_xlim()
        ymin, ymax = ax_xy.get_ylim()
        bbox = (xmin, xmax, ymin, ymax)

        if Axyk.ndim != 3 or Bk.ndim != 2 or Axyk.shape[0] != Bk.shape[0] or Axyk.shape[1] != Bk.shape[1] or Axyk.shape[2] != 2:
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

    def init():
        trail_line.set_data([], [])
        plan_line.set_data([], [])
        lidar_scatter.set_offsets(np.empty((0, 2)))
        ref_line.set_data([], [])
        ref_scatter.set_offsets(np.empty((0, 2)))

        _set_ship(ship_poly, float(px[0]), float(py[0]), float(psi[0]))
        _set_heading(float(px[0]), float(py[0]), float(psi[0]))
        time_text.set_text("")

        tl_line.set_data([], [])
        tr_line.set_data([], [])
        tb_line.set_data([], [])
        azl_line.set_data([], [])
        azr_line.set_data([], [])

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
            ship_poly,
            heading_line,
            time_text,
            tl_line,
            tr_line,
            tb_line,
            azl_line,
            azr_line,
        ]
        artists.extend(ghost_polys)
        artists.extend(col_polys)
        return tuple(artists)

    def update(frame_idx: int):
        k = int(frame_indices[frame_idx])

        trail_line.set_data(px[: k + 1], py[: k + 1])
        _set_ship(ship_poly, float(px[k]), float(py[k]), float(psi[k]))
        _set_heading(float(px[k]), float(py[k]), float(psi[k]))

        ghost_idxs = _set_plan_and_ghosts(step_k=k)
        _set_lidar(step_k=k)
        _set_reference(step_k=k)
        _set_collision_polys(step_k=k, ghost_idxs=ghost_idxs)

        tl_line.set_data(t[: k + 1], Tl_plot[: k + 1])
        tr_line.set_data(t[: k + 1], Tr_plot[: k + 1])
        tb_line.set_data(t[: k + 1], Tb_plot[: k + 1])
        azl_line.set_data(t[: k + 1], az_l_plot[: k + 1])
        azr_line.set_data(t[: k + 1], az_r_plot[: k + 1])

        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")

        artists = [
            trail_line,
            plan_line,
            ref_line,
            ref_scatter,
            lidar_scatter,
            ship_poly,
            heading_line,
            time_text,
            tl_line,
            tr_line,
            tb_line,
            azl_line,
            azr_line,
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

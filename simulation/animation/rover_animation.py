from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm


from core.models.simple_rover_model import SimpleRoverModel


def _resolve_results_dir(save_path: str | Path | None) -> Tuple[Path, Path]:
    """
    Returns (results_dir, mp4_path).

    If save_path is None:
        <project_root>/results/rover_animation.mp4
    If save_path is a directory:
        <save_path>/rover_animation.mp4
    If save_path is a file:
        that exact file path (and results_dir = parent)
    """
    if save_path is None:
        project_root = Path(__file__).resolve().parents[2]
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "rover_animation.mp4"

    save_path = Path(save_path)
    if save_path.is_dir():
        results_dir = save_path
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir, results_dir / "rover_animation.mp4"

    results_dir = save_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, save_path


def _infer_wheel_speed_bounds(system: SimpleRoverModel, constraints: object | None) -> Tuple[float, float]:
    """
    Infer bounds for wheel speeds omega_l, omega_r from state bounds:
      - we expect the augmented model: x = [px,py,phi,omega_l,omega_r]
      - and constraints to expose x_min/x_max (or get_bounds()).
    """
    if constraints is not None and hasattr(constraints, "get_bounds"):
        bounds = constraints.get_bounds()
        if len(bounds) == 4:
            x_min, x_max, _, _ = bounds
            x_min = np.asarray(x_min, dtype=float).reshape(-1)
            x_max = np.asarray(x_max, dtype=float).reshape(-1)
            if x_min.size >= 5 and x_max.size >= 5:
                wmin = float(min(x_min[3], x_min[4]))
                wmax = float(max(x_max[3], x_max[4]))
                return wmin, wmax

    x_min_attr = getattr(constraints, "x_min", None) if constraints is not None else None
    x_max_attr = getattr(constraints, "x_max", None) if constraints is not None else None
    if x_min_attr is not None and x_max_attr is not None:
        x_min = np.asarray(x_min_attr, dtype=float).reshape(-1)
        x_max = np.asarray(x_max_attr, dtype=float).reshape(-1)
        if x_min.size >= 5 and x_max.size >= 5:
            wmin = float(min(x_min[3], x_min[4]))
            wmax = float(max(x_max[3], x_max[4]))
            return wmin, wmax

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
    R = np.array([[c, -s], [s, c]], dtype=float)

    return (pts @ R.T) + np.array([px, py], dtype=float)


def _extract_pred_list(kwargs: dict) -> list[np.ndarray] | None:
    """
    Try multiple common names so main() can pass predictions without changing this file.
    Expected each element: array of shape (N+1, nx) for that control cycle.
    """
    for key in ("X_pred_traj", "X_pred_list", "X_preds", "X_horizon", "X_hist"):
        if key in kwargs and kwargs[key] is not None:
            pred = kwargs[key]
            if isinstance(pred, list):
                return pred
            arr = np.asarray(pred, dtype=float)
            # allow (T, N+1, nx) as a numpy array
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None

def _extract_ref_list(kwargs: dict) -> list[np.ndarray] | None:
    """
    Reference horizons per control cycle.
    Expected each element: array of shape (N+1, nx) for that cycle.
    """
    for key in ("X_ref_traj", "X_ref_list", "Xref_traj", "Xrefs"):
        if key in kwargs and kwargs[key] is not None:
            ref = kwargs[key]
            if isinstance(ref, list):
                return ref
            arr = np.asarray(ref, dtype=float)
            # allow (T, N+1, nx)
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def _extract_scan_list(kwargs: dict) -> list | None:
    """
    Accept a list of LaserScanLike objects OR a list/array of raw ranges arrays.
    """
    for key in ("lidar_scans", "scans", "scan_list"):
        if key in kwargs and kwargs[key] is not None:
            scans = kwargs[key]
            if isinstance(scans, list):
                return scans
            arr = np.asarray(scans)
            # allow (T, n_rays) raw ranges array
            if arr.ndim == 2:
                return [arr[i] for i in range(arr.shape[0])]
    return None


def _extract_global_path(kwargs: dict) -> np.ndarray | None:
    """
    Accept:
      - global_path: (M,2) or (M,>=2)
      - path_waypoints: same
    Returns (M,2) float or None.
    """
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

    - If scan has attribute `.ranges`: uses scan.angle_min / scan.angle_increment.
    - If scan is a 1D np.ndarray: assumes angles are evenly spaced [-pi, pi].

    Points with range=inf are ignored.
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
    """
    Rigid transform from robot frame to world frame:
      p_w = [px,py] + R(yaw) * p_r
    World: +x right, +y up.
    """
    if pts_r.size == 0:
        return pts_r.reshape(0, 2)

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    xw = px + c * pts_r[:, 0] - s * pts_r[:, 1]
    yw = py + s * pts_r[:, 0] + c * pts_r[:, 1]
    return np.column_stack([xw, yw])


def _maybe_load_occ_background(kwargs):
    """
    Returns (occ_img, extent) or (None, None).

    Supported kwargs:
      - occ_map: an OccupancyMap2D instance (preferred: avoids file I/O)
      - occ_cfg: an OccupancyMapConfig instance (will be loaded)
      - occ_map_path: str/Path to a map image (will be loaded)
      - occ_resolution, occ_origin, occ_threshold, occ_invert: params if occ_map_path is used

    Default map location (your repo layout):
      nav_mpc/simulation/environment/maps/map.png
    """
    try:
        from simulation.environment.occupancy_map import OccupancyMap2D, OccupancyMapConfig
    except Exception:
        return None, None

    occ_map = kwargs.get("occ_map", None)
    if occ_map is not None:
        # IMPORTANT: imshow(origin="lower") expects row 0 at bottom;
        # your occ_map.occ is in "row 0 is top" convention -> flip for display.
        occ = np.flipud(occ_map.occ).astype(float)
        extent = [occ_map.xmin, occ_map.xmax, occ_map.ymin, occ_map.ymax]
        return occ, extent

    occ_cfg = kwargs.get("occ_cfg", None)
    if occ_cfg is not None:
        occ_map = OccupancyMap2D.from_png(occ_cfg)
        occ = np.flipud(occ_map.occ).astype(float)
        extent = [occ_map.xmin, occ_map.xmax, occ_map.ymin, occ_map.ymax]
        return occ, extent

    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "simulation" / "environment" / "maps" / "map.png"

    occ_map_path = kwargs.get("occ_map_path", None)
    if occ_map_path is None:
        occ_map_path = default_path

    res = float(kwargs.get("occ_resolution", 0.05))
    origin = tuple(kwargs.get("occ_origin", (-10.0, -10.0)))
    thr = int(kwargs.get("occ_threshold", 127))
    inv = bool(kwargs.get("occ_invert", False))

    cfg = OccupancyMapConfig(
        map_path=str(occ_map_path),
        resolution=res,
        origin=origin,
        occupied_threshold=thr,
        invert=inv,
    )

    occ_map = OccupancyMap2D.from_png(cfg)
    occ = np.flipud(occ_map.occ).astype(float)
    extent = [occ_map.xmin, occ_map.xmax, occ_map.ymin, occ_map.ymax]
    return occ, extent


def animate_rover(
    system: SimpleRoverModel,
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
    What you will see:
      - occupancy map background (optional)
      - global path waypoints (optional) + thin polyline
      - robot trajectory
      - current plan + ghost poses (optional)
      - lidar points (optional)
      - wheel speeds
    """
    X_pred_list = _extract_pred_list(kwargs)
    X_ref_list  = _extract_ref_list(kwargs)
    scan_list = _extract_scan_list(kwargs)
    occ_img, occ_extent = _maybe_load_occ_background(kwargs)
    global_path = _extract_global_path(kwargs)

    TARGET_ANIM_FPS = 20.0
    VIDEO_DPI = 110
    GIF_DPI = 80

    N_GHOSTS = 5
    GHOST_ALPHA_MIN = 0.10
    PLAN_LINE_WIDTH = 2.0
    LIDAR_POINT_SIZE = 6
    OCC_ALPHA = 0.25

    # Global path style
    GP_LINE_WIDTH = 1.5
    GP_MARKER_SIZE = 16

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
    omega_l = x_arr[:, 3].astype(float)
    omega_r = x_arr[:, 4].astype(float)

    goal_xy = None
    if x_goal is not None:
        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        if x_goal.size >= 2:
            goal_xy = x_goal[:2].copy()

    L = float(getattr(system, "L"))
    half_side = float(L)

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
    fig.suptitle("Simple rover simulation")

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

    # Optional occupancy background (draw first so everything else overlays it)
    if occ_img is not None and occ_extent is not None:
        # 0 = free, 1 = occupied
        cmap = ListedColormap([
            "#ffefb0",  # free
            "#72498D",  # occupied
        ])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        ax_xy.imshow(
            occ_img,
            origin="lower",
            extent=occ_extent,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",  # smoother look (use "nearest" if you want crisp pixels)
            alpha=0.35,                # overall transparency of the map layer
            zorder=0,
        )

    # Global path (static)
    if global_path is not None and global_path.shape[0] >= 1:
        ax_xy.plot(global_path[:, 0], global_path[:, 1], linewidth=GP_LINE_WIDTH, linestyle="--", color="#ff7f0e", alpha=0.5, label="global path")
        ax_xy.scatter(global_path[:, 0], global_path[:, 1], s=GP_MARKER_SIZE, color="#ff7f0e", alpha=0.5, label="waypoints")

    # Start + goal
    ax_xy.plot(px[0], py[0], marker=".", markersize=3, linestyle="None", label="start")
    if goal_xy is not None:
        ax_xy.plot(goal_xy[0], goal_xy[1], marker="*", color="#ff7f0e", markersize=8, linestyle="None", label="goal")

    (trail_line,) = ax_xy.plot([], [], linestyle="--", color="#008000", linewidth=1.5, label="trajectory")
    (plan_line,) = ax_xy.plot([], [], linewidth=PLAN_LINE_WIDTH, color="#0058ca", linestyle="dotted", label="plan (current)")
    lidar_scatter = ax_xy.scatter([], [], s=LIDAR_POINT_SIZE, marker=".", color="#b11010", alpha=0.8, label="lidar")
    (ref_line,) = ax_xy.plot([], [], linewidth=2.0, linestyle="-", color="#72498D", alpha=0.1, label="reference (current)")
    ref_scatter = ax_xy.scatter([], [], s=25, marker="o", color="#72498D", alpha=0.1, label="ref points")

    body_poly = Polygon(
        _square_body_vertices(float(px[0]), float(py[0]), float(phi[0]), half_side),
        closed=True,
        facecolor='#1f77b4',
        edgecolor='#1f77b4',
        linewidth=0.1,
        alpha=1.0,            
        zorder=6,
    )
    ax_xy.add_patch(body_poly)

    ghost_polys: list[Polygon] = []
    if X_pred_list is not None:
        ghost_alphas = np.linspace(1.0, GHOST_ALPHA_MIN, N_GHOSTS + 1)[1:]
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
    ax_xy.legend(loc="best")
    ax_xy.get_legend().set_visible(False)

    # -------- wheel speed axis --------
    ax_w.set_title("Wheel speeds")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.grid(True)
    ax_w.set_xlim(float(t[0]), float(t[-1]))
    ax_w.set_ylim(-1.1 * wabs, 1.1 * wabs)
    ax_w.axhline(0.0, color="k", linewidth=0.8, alpha=0.8)
    ax_w.axhline(float(wmin), linestyle="--", linewidth=1.5, color="#b11010", alpha=0.3)
    ax_w.axhline(float(wmax), linestyle="--", linewidth=1.5, color="#b11010", alpha=0.3)

    (w_l_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_l$")
    (w_r_line,) = ax_w.plot([], [], linewidth=2, label=r"$\omega_r$")
    ax_w.legend(loc="best")

    def _set_body(poly: Polygon, px_k: float, py_k: float, phi_k: float) -> None:
        poly.set_xy(_square_body_vertices(px_k, py_k, phi_k, half_side))

    def _set_heading(px_k: float, py_k: float, phi_k: float) -> None:
        hlen = 1.2 * half_side
        xh = px_k + hlen * float(np.cos(phi_k))
        yh = py_k + hlen * float(np.sin(phi_k))
        heading_line.set_data([px_k, xh], [py_k, yh])

    def _set_plan_and_ghosts(step_k: int) -> None:
        if X_pred_list is None:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_visible(False)
            return

        last_cycle = min(max(step_k, 0), len(X_pred_list) - 1)
        pred = np.asarray(X_pred_list[last_cycle], dtype=float)

        if pred.ndim != 2 or pred.shape[1] < 3:
            plan_line.set_data([], [])
            for gp in ghost_polys:
                gp.set_visible(False)
            return

        plan_line.set_data(pred[:, 0], pred[:, 1])

        Nh = pred.shape[0] - 1
        if Nh <= 0:
            for gp in ghost_polys:
                gp.set_visible(False)
            return

        idxs = np.linspace(0, Nh, N_GHOSTS + 1).round().astype(int)
        ghost_idxs = idxs[1:]

        for gp, ii in zip(ghost_polys, ghost_idxs, strict=False):
            _set_body(gp, float(pred[ii, 0]), float(pred[ii, 1]), float(_wrap_to_pi(np.array([pred[ii, 2]])).item()))
            gp.set_visible(True)

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

        artists = [trail_line, plan_line, ref_line, ref_scatter, lidar_scatter, body_poly, heading_line, time_text, w_l_line, w_r_line]
        artists.extend(ghost_polys)
        return tuple(artists)

    def update(frame_idx: int):
        k = int(frame_indices[frame_idx])

        trail_line.set_data(px[: k + 1], py[: k + 1])
        _set_body(body_poly, float(px[k]), float(py[k]), float(phi[k]))
        _set_heading(float(px[k]), float(py[k]), float(phi[k]))

        _set_plan_and_ghosts(step_k=k)
        _set_lidar(step_k=k)
        _set_reference(step_k=k)

        w_l_line.set_data(t[: k + 1], omega_l[: k + 1])
        w_r_line.set_data(t[: k + 1], omega_r[: k + 1])
        time_text.set_text(f"t = {t_anim[frame_idx]:.2f} s")

        artists = [trail_line, plan_line, ref_line, ref_scatter, lidar_scatter, body_poly, heading_line, time_text, w_l_line, w_r_line]
        artists.extend(ghost_polys)
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

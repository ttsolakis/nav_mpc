# nav_mpc/simulation/lidar/lidar_simulator.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from simulation.environment.occupancy_map import OccupancyMap2D


@dataclass(frozen=True, slots=True)
class LaserScanLike:
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray  # shape (n_rays,)


@dataclass(frozen=True, slots=True)
class LidarConfig:
    # ROS-like angular layout
    angle_min: float = -np.pi
    angle_max: float = np.pi
    angle_increment: float = np.deg2rad(0.72)  # ~500 rays over 360deg

    # range limits
    range_min: float = 0.05
    range_max: float = 8.0

    # ray-marching resolution (meters). If None, uses 0.5 * map.resolution.
    ray_step: float | None = None

    # deterministic noise/dropout (optional)
    noise_std: float = 0.0
    drop_prob: float = 0.0
    seed: int = 0


class LidarSimulator2D:
    """
    2D lidar ray-caster against an OccupancyMap2D.

    Pose input is (x, y, yaw) in WORLD frame.
    Scan angles are in the robot/lidar frame:
      angle=0 -> +x forward, angle=+pi/2 -> +y left
    """

    def __init__(self, occ_map: OccupancyMap2D, cfg: LidarConfig):
        self.map = occ_map
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        n = int(np.floor((cfg.angle_max - cfg.angle_min) / cfg.angle_increment)) + 1
        self.angles = cfg.angle_min + np.arange(n) * cfg.angle_increment

        if cfg.ray_step is None:
            self.ray_step = 0.5 * self.map.res
        else:
            self.ray_step = float(cfg.ray_step)
            if self.ray_step <= 0:
                raise ValueError("ray_step must be > 0")

    def scan(self, pose_xy_yaw: np.ndarray) -> LaserScanLike:
        x, y, yaw = float(pose_xy_yaw[0]), float(pose_xy_yaw[1]), float(pose_xy_yaw[2])

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        ranges = np.empty(self.angles.shape[0], dtype=float)

        for i, a in enumerate(self.angles):
            # direction in world = R(yaw) * [cos(a), sin(a)]
            ca = np.cos(a)
            sa = np.sin(a)
            dx = cy * ca - sy * sa
            dy = sy * ca + cy * sa

            r = self._raycast(x, y, dx, dy)

            # dropout/noise (optional)
            if self.cfg.drop_prob > 0.0 and self.rng.random() < self.cfg.drop_prob:
                r = np.inf
            elif self.cfg.noise_std > 0.0 and np.isfinite(r):
                r = r + float(self.rng.normal(0.0, self.cfg.noise_std))

            # clamp to ROS-like rules
            if (not np.isfinite(r)) or (r < self.cfg.range_min):
                ranges[i] = np.inf
            else:
                ranges[i] = float(min(r, self.cfg.range_max))

        return LaserScanLike(
            angle_min=self.cfg.angle_min,
            angle_max=self.cfg.angle_max,
            angle_increment=self.cfg.angle_increment,
            range_min=self.cfg.range_min,
            range_max=self.cfg.range_max,
            ranges=ranges,
        )
    
    def get_angles(self) -> np.ndarray:
        """Angles of the rays in the ROBOT/LIDAR frame (0 = forward, +pi/2 = left)."""
        return self.angles.copy()

    def scan_polar(self, pose_xy_yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (angles, ranges) where:
          angles: (n_rays,) in robot frame
          ranges: (n_rays,) in meters, inf where no return (ROS-like)
        """
        scan = self.scan(pose_xy_yaw)
        return self.angles.copy(), scan.ranges.copy()

    def scan_points_robot(self, pose_xy_yaw: np.ndarray, *, finite_only: bool = True) -> np.ndarray:
        """
        Returns lidar hit points in the ROBOT frame as (M,2).

        If finite_only=True: drops inf ranges (no hit).
        """
        scan = self.scan(pose_xy_yaw)
        ranges = np.asarray(scan.ranges, dtype=float).reshape(-1)
        angles = self.angles  # already robot-frame angles

        if finite_only:
            mask = np.isfinite(ranges)
            ranges = ranges[mask]
            angles = angles[mask]

        xr = ranges * np.cos(angles)
        yr = ranges * np.sin(angles)
        return np.column_stack([xr, yr])
    
    def scan_points_world(self, pose_xy_yaw: np.ndarray, *, finite_only: bool = True) -> np.ndarray:
        """
        Returns lidar hit points in the WORLD frame as (M,2).

        If finite_only=True: drops inf ranges (no hit).
        """
        x, y, yaw = float(pose_xy_yaw[0]), float(pose_xy_yaw[1]), float(pose_xy_yaw[2])
        scan = self.scan(pose_xy_yaw)

        ranges = np.asarray(scan.ranges, dtype=float).reshape(-1)
        angles = self.angles  # robot-frame ray angles

        if finite_only:
            mask = np.isfinite(ranges)
            ranges = ranges[mask]
            angles = angles[mask]

        # points in robot frame
        xr = ranges * np.cos(angles)
        yr = ranges * np.sin(angles)

        # transform to world
        c = np.cos(yaw)
        s = np.sin(yaw)
        xw = x + c * xr - s * yr
        yw = y + s * xr + c * yr

        return np.column_stack([xw, yw])
    
    def points_world_from_scan(
        self,
        scan,
        pose_xy_yaw: np.ndarray,
        *,
        finite_only: bool = True,
    ) -> np.ndarray:
        """
        Convert an existing LaserScanLike to WORLD-frame hit points (M,2).
        No raycasting is performed here.
        """
        x, y, yaw = float(pose_xy_yaw[0]), float(pose_xy_yaw[1]), float(pose_xy_yaw[2])

        ranges = np.asarray(scan.ranges, dtype=float).reshape(-1)
        angles = self.angles  # robot-frame angles

        if finite_only:
            mask = np.isfinite(ranges)
            ranges = ranges[mask]
            angles = angles[mask]

        # robot-frame points
        xr = ranges * np.cos(angles)
        yr = ranges * np.sin(angles)

        # robot -> world
        c = np.cos(yaw)
        s = np.sin(yaw)
        xw = x + c * xr - s * yr
        yw = y + s * xr + c * yr

        return np.column_stack([xw, yw])



    def _raycast(self, x0: float, y0: float, dx: float, dy: float) -> float:
        """
        March along the ray until we hit an occupied cell or exceed range_max.
        Returns hit distance in meters, or range_max if none.
        """
        max_r = self.cfg.range_max
        step = self.ray_step

        # If starting inside obstacle, return 0 (will be converted to inf by range_min rule)
        if self.map.is_occupied_world(x0, y0):
            return 0.0

        r = 0.0
        while r <= max_r:
            x = x0 + r * dx
            y = y0 + r * dy
            if self.map.is_occupied_world(x, y):
                return r
            r += step

        return max_r

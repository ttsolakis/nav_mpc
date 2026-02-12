# core/augmentations/arc_length_path.py
from __future__ import annotations

import numpy as np


class ArcLengthPath:
    """
    Polyline path parameterized by arc length s (meters).

    Given points P[i] = (x_i, y_i), we precompute cumulative arc length S[i].
    Then:
      - pos(s): linear interpolation along the segment containing s
      - tangent(s): unit tangent of that segment (piecewise constant)
    """

    def __init__(self, points_xy: np.ndarray):
        pts = np.asarray(points_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            raise ValueError(f"points_xy must be (M,2) with M>=2, got {pts.shape}")

        d = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(d, axis=1)
        if np.any(seg_len < 1e-12):
            # allow, but it makes indexing tricky — remove duplicates
            keep = np.ones(pts.shape[0], dtype=bool)
            keep[1:] = seg_len >= 1e-12
            pts = pts[keep]
            d = pts[1:] - pts[:-1]
            seg_len = np.linalg.norm(d, axis=1)
            if pts.shape[0] < 2:
                raise ValueError("Path became degenerate after removing duplicates.")

        self._pts = pts
        self._seg = d
        self._seg_len = seg_len
        self._S = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._L = float(self._S[-1])

    @property
    def length(self) -> float:
        return self._L

    @property
    def points(self) -> np.ndarray:
        return self._pts

    def _clamp_s(self, s: float) -> float:
        return float(np.clip(s, 0.0, self._L))

    def _segment_index(self, s: float) -> int:
        # find i such that S[i] <= s < S[i+1]
        # np.searchsorted returns insertion index
        i = int(np.searchsorted(self._S, s, side="right") - 1)
        return int(np.clip(i, 0, self._seg_len.size - 1))

    def pos(self, s: float) -> np.ndarray:
        s = self._clamp_s(s)
        if self._L < 1e-12:
            return self._pts[0].copy()

        i = self._segment_index(s)
        s0 = self._S[i]
        ds = s - s0
        Ls = self._seg_len[i]
        a = 0.0 if Ls < 1e-12 else ds / Ls
        return (1.0 - a) * self._pts[i] + a * self._pts[i + 1]

    def tangent(self, s: float) -> np.ndarray:
        s = self._clamp_s(s)
        if self._L < 1e-12:
            return np.array([1.0, 0.0], dtype=float)

        i = self._segment_index(s)
        v = self._seg[i]
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    def heading(self, s: float) -> float:
        t = self.tangent(s)
        return float(np.arctan2(t[1], t[0]))

    def project(self, p: np.ndarray, s_hint: float | None = None, window_m: float | None = None) -> float:
        """
        Project point p onto the polyline and return arc-length coordinate s_proj.

        Optional:
          - s_hint: search around a hint for speed
          - window_m: search only segments whose arc-length interval intersects [s_hint-window_m, s_hint+window_m]
        """
        p = np.asarray(p, dtype=float).reshape(2)
        if self._seg_len.size == 0:
            return 0.0

        seg_start_idx = 0
        seg_end_idx = self._seg_len.size

        if s_hint is not None and window_m is not None and window_m > 0.0:
            s_hint = self._clamp_s(float(s_hint))
            lo = max(0.0, s_hint - float(window_m))
            hi = min(self._L, s_hint + float(window_m))
            i_lo = int(np.searchsorted(self._S, lo, side="right") - 1)
            i_hi = int(np.searchsorted(self._S, hi, side="left"))
            seg_start_idx = int(np.clip(i_lo, 0, self._seg_len.size - 1))
            seg_end_idx = int(np.clip(i_hi, seg_start_idx + 1, self._seg_len.size))

        best_d2 = np.inf
        best_s = 0.0

        for i in range(seg_start_idx, seg_end_idx):
            a = self._pts[i]
            b = self._pts[i + 1]
            ab = b - a
            L2 = float(np.dot(ab, ab))
            if L2 < 1e-12:
                continue
            t = float(np.dot(p - a, ab) / L2)
            t = float(np.clip(t, 0.0, 1.0))
            q = a + t * ab
            d2 = float(np.dot(p - q, p - q))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self._S[i] + t * self._seg_len[i])

        return self._clamp_s(best_s)

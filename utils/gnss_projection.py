from __future__ import annotations

import configparser
import math
import time
from pathlib import Path

import cv2
import numpy as np

from utils.realtime_navigation import RealtimeNavigationState


class CalibratedGroundProjector:
    def __init__(self, ini_path: str | Path, width: int, height: int, ground_z: float):
        cfg = configparser.ConfigParser()
        if not cfg.read(ini_path, encoding="utf-8"):
            raise FileNotFoundError(ini_path)
        sx = width / cfg.getfloat("calibration", "width")
        sy = height / cfg.getfloat("calibration", "height")
        self.camera_matrix = np.array(
            [
                [
                    cfg.getfloat("calibration", "fx") * sx,
                    0.0,
                    cfg.getfloat("calibration", "cx") * sx,
                ],
                [
                    0.0,
                    cfg.getfloat("calibration", "fy") * sy,
                    cfg.getfloat("calibration", "cy") * sy,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        extrinsic = np.asarray(
            [
                [
                    float(value)
                    for value in cfg.get(
                        "calibration", f"extrinsic_row{index}"
                    ).split(",")
                ]
                for index in range(4)
            ],
            dtype=np.float64,
        )
        self.rotation = extrinsic[:3, :3]
        self.translation = extrinsic[:3, 3]
        self.rotation_vector, _ = cv2.Rodrigues(self.rotation)
        self.distortion = np.array(
            [
                cfg.getfloat("distortion", key)
                for key in ("k1", "k2", "p1", "p2", "k3")
            ]
        )
        self.width = int(width)
        self.height = int(height)
        self.ground_z = float(ground_z)

    def project(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points_xy) == 0:
            return np.empty((0, 2), np.float32), np.empty(0, dtype=bool)
        points_3d = np.column_stack(
            (
                points_xy[:, 0],
                points_xy[:, 1],
                np.full(len(points_xy), self.ground_z),
            )
        ).astype(np.float64)
        camera_points = points_3d @ self.rotation.T + self.translation
        pixels, _ = cv2.projectPoints(
            points_3d,
            self.rotation_vector,
            self.translation,
            self.camera_matrix,
            self.distortion,
        )
        pixels = pixels.reshape(-1, 2)
        valid = (
            (camera_points[:, 2] > 0.05)
            & np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] > -8 * self.width)
            & (pixels[:, 0] < 9 * self.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.height)
        )
        return pixels, valid


def _project_centerline(
    projector: CalibratedGroundProjector,
    body_path: np.ndarray,
) -> np.ndarray:
    pixels, valid = projector.project(body_path)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return np.empty((0, 2), dtype=np.int32)
    end = 1
    while end < len(indices) and indices[end] == indices[end - 1] + 1:
        end += 1
    if end < 2:
        return np.empty((0, 2), dtype=np.int32)
    curve = np.rint(pixels[indices[:end]]).astype(np.int32)
    offset = curve[0].astype(np.float64) - np.array(
        [projector.width / 2.0, projector.height - 1.0]
    )
    return np.rint(curve.astype(np.float64) - offset).astype(np.int32)


class NavigationPathSmoother:
    def __init__(
        self,
        temporal_alpha: float = 0.90,
        spatial_window: int = 11,
        near_preserve_m: float = 4.0,
        near_transition_m: float = 4.0,
    ):
        self.temporal_alpha = float(temporal_alpha)
        self.spatial_window = max(3, int(spatial_window) | 1)
        self.near_preserve_m = max(0.0, float(near_preserve_m))
        self.near_transition_m = max(0.0, float(near_transition_m))
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def _smoothing_strength(self, path: np.ndarray) -> np.ndarray:
        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
        )
        if self.near_transition_m <= 1e-6:
            strength = (distance > self.near_preserve_m).astype(np.float64)
        else:
            strength = np.clip(
                (distance - self.near_preserve_m) / self.near_transition_m,
                0.0,
                1.0,
            )
            strength = strength * strength * (3.0 - 2.0 * strength)
        strength[0] = 0.0
        return strength[:, None]

    def _spatial(self, path: np.ndarray) -> np.ndarray:
        if len(path) < 3:
            return path.astype(np.float32, copy=True)
        source = path.astype(np.float64)
        kernel = np.convolve(
            np.ones(self.spatial_window), np.ones(self.spatial_window)
        )
        kernel /= kernel.sum()
        radius = len(kernel) // 2
        padded = np.pad(
            source, ((radius, radius), (0, 0)), mode="edge"
        )
        smoothed = np.column_stack(
            [
                np.convolve(padded[:, axis], kernel, mode="valid")
                for axis in range(2)
            ]
        )
        strength = self._smoothing_strength(source)
        blended = source + strength * (smoothed - source)
        blended[0] = source[0]
        return blended.astype(np.float32)

    @staticmethod
    def _resample(path: np.ndarray, normalized_distance: np.ndarray) -> np.ndarray:
        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
        )
        if distance[-1] < 1e-6:
            return np.repeat(path[:1], len(normalized_distance), axis=0)
        distance /= distance[-1]
        return np.column_stack(
            [
                np.interp(normalized_distance, distance, path[:, axis])
                for axis in range(2)
            ]
        )

    def update(self, path: np.ndarray | None) -> np.ndarray | None:
        if path is None or len(path) < 2:
            self.reset()
            return None
        current = self._spatial(path)
        if self.previous is not None:
            distance = np.concatenate(
                (
                    [0.0],
                    np.cumsum(
                        np.hypot(np.diff(current[:, 0]), np.diff(current[:, 1]))
                    ),
                )
            )
            if distance[-1] > 1e-6:
                previous = self._resample(self.previous, distance / distance[-1])
                strength = self._smoothing_strength(current)
                current = current + strength * self.temporal_alpha * (
                    previous - current
                )
        current[0] = [0.0, 0.0]
        self.previous = current.astype(np.float32)
        return self.previous.copy()


class RealtimeGnssProjectionEngine:
    """Project the thread-safe realtime navigation state into video pixels."""

    def __init__(
        self,
        *,
        navigation_state: RealtimeNavigationState,
        calibration_path: str,
        width: int,
        height: int,
        lidar_height_m: float,
        projection_scale: float,
        update_hz: float = 10.0,
        smooth_alpha: float = 0.90,
        smooth_window: int = 11,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Projection dimensions must be greater than zero")
        if not 0.0 < projection_scale <= 1.0:
            raise ValueError("nav-scale must be in (0, 1]")
        if not math.isfinite(update_hz) or update_hz <= 0.0:
            raise ValueError("GNSS projection update rate must be greater than zero")
        self.navigation_state = navigation_state
        self.width = int(width)
        self.height = int(height)
        self.projection_scale = float(projection_scale)
        self.update_interval_s = 1.0 / float(update_hz)
        self.projector = CalibratedGroundProjector(
            calibration_path,
            self.width,
            self.height,
            ground_z=-abs(float(lidar_height_m)),
        )
        effective_alpha = float(smooth_alpha) ** (30.0 / float(update_hz))
        self.path_smoother = NavigationPathSmoother(
            temporal_alpha=effective_alpha,
            spatial_window=smooth_window,
        )
        self._last_projection_monotonic: float | None = None
        self._previous_curve = np.empty((0, 2), dtype=np.int32)

        initial = self._project_body_path(self.navigation_state.body_path())
        if len(initial) < 2:
            raise ValueError("Calibration cannot project the straight navigation band")
        self._previous_curve = initial

    def _scale_about_bottom_center(self, curve: np.ndarray) -> np.ndarray:
        if len(curve) == 0 or self.projection_scale == 1.0:
            return curve
        anchor = np.array(
            [self.width / 2.0, self.height - 1.0], dtype=np.float64
        )
        scaled = anchor + (curve.astype(np.float64) - anchor) * self.projection_scale
        return np.rint(scaled).astype(np.int32)

    def _project_body_path(self, body_path: np.ndarray) -> np.ndarray:
        center_px = _project_centerline(self.projector, body_path)
        return self._scale_about_bottom_center(center_px)

    def project(
        self,
        now_epoch: float | None = None,
        now_monotonic: float | None = None,
    ) -> np.ndarray:
        monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        body_path = self.navigation_state.body_path(
            now_epoch=now_epoch,
            now_monotonic=monotonic,
        )
        if (
            self._last_projection_monotonic is not None
            and monotonic - self._last_projection_monotonic < self.update_interval_s
        ):
            return self._previous_curve.copy()
        self._last_projection_monotonic = monotonic

        smoothed = self.path_smoother.update(body_path)
        if smoothed is None:
            return self._previous_curve.copy()
        center_px = self._project_body_path(smoothed)
        if len(center_px) < 2:
            return self._previous_curve.copy()
        self._previous_curve = center_px
        return center_px.copy()

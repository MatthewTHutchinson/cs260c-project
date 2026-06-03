"""Frame-convention helpers for simulator adapters.

The algorithm avoids world pose, but adapters still need explicit transforms
when reading simulator telemetry or comparing logs. Keep those transforms
centralized so ENU/NED bugs are not scattered across the codebase.
"""

from __future__ import annotations

import numpy as np


def enu_position_to_ned(position_enu: np.ndarray) -> np.ndarray:
    """Convert ENU `[east, north, up]` to NED `[north, east, down]`."""
    east, north, up = np.asarray(position_enu, dtype=np.float64)[:3]
    return np.array([north, east, -up], dtype=np.float64)


def ned_position_to_enu(position_ned: np.ndarray) -> np.ndarray:
    """Convert NED `[north, east, down]` to ENU `[east, north, up]`."""
    north, east, down = np.asarray(position_ned, dtype=np.float64)[:3]
    return np.array([east, north, -down], dtype=np.float64)


def enu_vector_to_ned(vector_enu: np.ndarray) -> np.ndarray:
    """Convert a free vector from ENU to NED."""
    return enu_position_to_ned(vector_enu)


def ned_vector_to_enu(vector_ned: np.ndarray) -> np.ndarray:
    """Convert a free vector from NED to ENU."""
    return ned_position_to_enu(vector_ned)


def body_forward_elevation_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Return world elevation of the body +X axis from an `[x, y, z, w]` quat.

    Elodin exposes orientation as the quaternion portion of `world_pos`. The
    algorithm may use attitude/orientation telemetry, but not global position.
    This helper extracts only the body-forward pitch/elevation cue needed to
    interpret a tilted FPV camera.
    """
    qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=np.float64)[:4]
    norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if norm <= 1e-9:
        return 0.0
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    forward_z = 2.0 * (qx * qz - qy * qw)
    return float(np.arcsin(np.clip(forward_z, -1.0, 1.0)))

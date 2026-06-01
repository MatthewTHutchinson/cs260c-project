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

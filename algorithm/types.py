"""Small data types for the VQ1 autonomous racing algorithm.

These types intentionally exclude global/world pose. Simulator backends may
have pose for scoring or logging, but the competition-facing algorithm should
only consume FPV-derived gate estimates plus allowed telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class TrackMode(str, Enum):
    """Navigation-facing confidence state for the next gate estimate."""

    DETECTED = "detected"
    TRACKED = "tracked"
    SEARCH = "search"
    COMMIT = "commit"
    RECOVER = "recover"


@dataclass(frozen=True)
class GateEstimate:
    """FPV-derived estimate of the next gate.

    Bearings use image/camera convention:
    - `bearing_h_rad`: positive means gate center appears right of image center.
    - `bearing_v_rad`: positive means gate center appears above image center.
    """

    bearing_h_rad: float = 0.0
    bearing_v_rad: float = 0.0
    distance_m: Optional[float] = None
    confidence: float = 0.0
    pixel_center: Optional[tuple[float, float]] = None
    apparent_size_px: Optional[float] = None
    sequence_index: int = 0
    age_s: float = 0.0
    mode: TrackMode = TrackMode.SEARCH

    @property
    def has_range(self) -> bool:
        return self.distance_m is not None and self.distance_m > 0.0

    @property
    def is_usable(self) -> bool:
        return self.mode in {TrackMode.DETECTED, TrackMode.TRACKED, TrackMode.COMMIT}


@dataclass
class VehicleTelemetry:
    """Allowed vehicle telemetry consumed by the racing algorithm."""

    attitude_quat: Optional[np.ndarray] = None
    rpy_rad: Optional[np.ndarray] = None
    angular_rates_rad_s: Optional[np.ndarray] = None
    linear_velocity_m_s: Optional[np.ndarray] = None
    acceleration_m_s2: Optional[np.ndarray] = None
    timestamp_s: float = 0.0
    status_flags: Optional[dict] = None


@dataclass(frozen=True)
class RacingCommand:
    """Internal body-rate/thrust command.

    This is the algorithm boundary. Deployment code maps it to MAVSDK
    attitude-rate/thrust, Elodin/Betaflight RC commands, or other runtime
    adapters.
    """

    roll_rate_rad_s: float = 0.0
    pitch_rate_rad_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    thrust_norm: float = 0.5
    mode: TrackMode = TrackMode.SEARCH

    def clipped(
        self,
        max_roll_rate: float,
        max_pitch_rate: float,
        max_yaw_rate: float,
        min_thrust: float,
        max_thrust: float,
    ) -> "RacingCommand":
        return RacingCommand(
            roll_rate_rad_s=float(np.clip(self.roll_rate_rad_s, -max_roll_rate, max_roll_rate)),
            pitch_rate_rad_s=float(np.clip(self.pitch_rate_rad_s, -max_pitch_rate, max_pitch_rate)),
            yaw_rate_rad_s=float(np.clip(self.yaw_rate_rad_s, -max_yaw_rate, max_yaw_rate)),
            thrust_norm=float(np.clip(self.thrust_norm, min_thrust, max_thrust)),
            mode=self.mode,
        )

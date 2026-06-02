"""Adapters from the internal racing command to backend-specific fields."""

from __future__ import annotations

import numpy as np

from algorithm.types import RacingCommand


def to_mavsdk_attitude_rate_fields(command: RacingCommand) -> dict[str, float]:
    """Return explicit fields for a MAVSDK attitude-rate/thrust call.

    This deliberately returns plain data instead of importing MAVSDK. The VQ1
    runtime adapter should own the concrete SDK object and the final sign probe.
    """
    return {
        "roll_rate_rad_s": float(command.roll_rate_rad_s),
        "pitch_rate_rad_s": float(command.pitch_rate_rad_s),
        "yaw_rate_rad_s": float(command.yaw_rate_rad_s),
        "thrust_norm": float(np.clip(command.thrust_norm, 0.0, 1.0)),
    }


def to_betaflight_rc_fields(
    command: RacingCommand,
    *,
    roll_scale: float = 350.0,
    pitch_scale: float = 350.0,
    yaw_scale: float = 350.0,
    throttle_scale: float = 700.0,
    arm: bool = True,
) -> dict[str, int]:
    """Map a RacingCommand to generic Betaflight-style RC PWM fields.

    This returns plain channel values so the Elodin solver can construct its
    local `RCCommand` without making this repo depend on Elodin's package.
    The signs still need final probing in the harness.
    """
    throttle = 1000.0 + np.clip(command.thrust_norm, 0.0, 1.0) * throttle_scale
    roll = 1500.0 + command.roll_rate_rad_s * roll_scale
    # Betaflight RC pitch is opposite our internal body-rate convention:
    # internal negative pitch accelerates forward, while RC values above
    # center command the forward/nose-down direction in this harness.
    pitch = 1500.0 - command.pitch_rate_rad_s * pitch_scale
    yaw = 1500.0 + command.yaw_rate_rad_s * yaw_scale

    return {
        "throttle": int(round(np.clip(throttle, 1000, 2000))),
        "roll": int(round(np.clip(roll, 1000, 2000))),
        "pitch": int(round(np.clip(pitch, 1000, 2000))),
        "yaw": int(round(np.clip(yaw, 1000, 2000))),
        "arm": 1800 if arm else 1000,
    }

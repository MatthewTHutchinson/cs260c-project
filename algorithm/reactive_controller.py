"""Reactive gate-following controller for completion-first VQ1 racing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from algorithm.frames import body_forward_elevation_from_quat_xyzw
from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry


@dataclass(frozen=True)
class ReactiveControllerGains:
    """Controller gains and safety limits."""

    yaw_gain: float = 2.2
    roll_gain: float = 0.35
    vertical_gain: float = 0.75
    forward_gain: float = 0.24
    damping_pitch_from_speed: float = 0.04
    hover_thrust: float = 0.52
    min_thrust: float = 0.30
    max_thrust: float = 0.95
    search_settle_s: float = 0.75
    search_yaw_rate_rad_s: float = 0.45
    search_pitch_level_gain: float = 0.90
    search_max_pitch_level_rate_rad_s: float = 0.35
    search_level_tolerance_rad: float = 0.05
    search_forward_velocity_damping: float = 0.07
    search_lateral_velocity_damping: float = 0.18
    search_max_velocity_brake_rate_rad_s: float = 0.25
    search_velocity_settle_m_s: float = 0.45
    max_roll_rate_rad_s: float = 0.70
    max_pitch_rate_rad_s: float = 0.80
    max_yaw_rate_rad_s: float = 1.20
    target_pass_distance_m: float = 1.4
    far_distance_m: float = 5.5
    minimum_track_confidence: float = 0.0
    vertical_forward_deadband_rad: float = 0.02
    vertical_forward_suppression_rad: float = 0.16
    camera_tilt_up_rad: float = math.radians(20.0)
    commit_lateral_scale: float = 0.25
    commit_yaw_scale: float = 0.25
    commit_forward_scale: float = 1.15


class ReactiveGateController:
    """Map a tracked gate estimate into body-rate/thrust commands."""

    def __init__(self, gains: ReactiveControllerGains | None = None) -> None:
        self.gains = gains or ReactiveControllerGains()
        self._search_started_s: float | None = None

    def reset(self) -> None:
        self._search_started_s = None

    def compute(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry | None = None,
    ) -> RacingCommand:
        telemetry = telemetry or VehicleTelemetry()

        if not gate.is_usable or gate.confidence < self.gains.minimum_track_confidence:
            return self._search_command(telemetry)

        self._search_started_s = None

        confidence_scale = float(np.clip(gate.confidence, 0.15, 1.0))
        centered_scale = float(np.clip(1.0 - abs(gate.bearing_h_rad) / 0.9, 0.25, 1.0))

        yaw_rate = self.gains.yaw_gain * gate.bearing_h_rad * confidence_scale
        roll_rate = self.gains.roll_gain * gate.bearing_h_rad * confidence_scale
        body_elevation_rad = self._body_vertical_bearing(gate)
        thrust = self.gains.hover_thrust + self.gains.vertical_gain * body_elevation_rad

        pitch_rate = self._forward_pitch_rate(gate, telemetry, confidence_scale, centered_scale)

        if gate.mode == TrackMode.COMMIT:
            # Close, partially clipped gate contours can pull the apparent
            # target toward an edge. Commit mostly preserves pass-through
            # motion instead of chasing a corner-shaped visual residual.
            pitch_rate *= self.gains.commit_forward_scale
            roll_rate *= self.gains.commit_lateral_scale
            yaw_rate *= self.gains.commit_yaw_scale

        return RacingCommand(
            roll_rate_rad_s=roll_rate,
            pitch_rate_rad_s=pitch_rate,
            yaw_rate_rad_s=yaw_rate,
            thrust_norm=thrust,
            mode=gate.mode,
        ).clipped(
            self.gains.max_roll_rate_rad_s,
            self.gains.max_pitch_rate_rad_s,
            self.gains.max_yaw_rate_rad_s,
            self.gains.min_thrust,
            self.gains.max_thrust,
        )

    def _search_command(self, telemetry: VehicleTelemetry) -> RacingCommand:
        now_s = self._timestamp_s(telemetry)
        if self._search_started_s is None:
            self._search_started_s = now_s
        search_elapsed_s = max(0.0, now_s - self._search_started_s)

        roll_brake_rate, pitch_brake_rate = self._search_velocity_brake_rates(telemetry)
        is_settled = self._search_velocity_is_settled(telemetry)

        # Prioritize braking: only apply leveling if the drone has settled.
        if is_settled:
            pitch_level_rate = self._search_pitch_level_rate(telemetry)
            pitch_rate = pitch_level_rate + pitch_brake_rate
        else:
            pitch_rate = pitch_brake_rate

        is_level = (
            abs(self._body_forward_elevation(telemetry))
            <= self.gains.search_level_tolerance_rad
        )
        yaw_rate = (
            self.gains.search_yaw_rate_rad_s
            if search_elapsed_s >= self.gains.search_settle_s and is_level and is_settled
            else 0.0
        )

        return RacingCommand(
            roll_rate_rad_s=roll_brake_rate,
            pitch_rate_rad_s=pitch_rate,
            yaw_rate_rad_s=yaw_rate,
            thrust_norm=self.gains.hover_thrust,
            mode=TrackMode.SEARCH,
        ).clipped(
            self.gains.max_roll_rate_rad_s,
            self.gains.max_pitch_rate_rad_s,
            self.gains.max_yaw_rate_rad_s,
            self.gains.min_thrust,
            self.gains.max_thrust,
        )

    def _search_pitch_level_rate(self, telemetry: VehicleTelemetry) -> float:
        body_elevation = self._body_forward_elevation(telemetry)
        if abs(body_elevation) <= self.gains.search_level_tolerance_rad:
            return 0.0
        return float(
            np.clip(
                -self.gains.search_pitch_level_gain * body_elevation,
                -self.gains.search_max_pitch_level_rate_rad_s,
                self.gains.search_max_pitch_level_rate_rad_s,
            )
        )

    def _search_velocity_brake_rates(
        self,
        telemetry: VehicleTelemetry,
    ) -> tuple[float, float]:
        body_velocity = telemetry.linear_velocity_m_s
        if body_velocity is None:
            return 0.0, 0.0

        velocity = np.asarray(body_velocity, dtype=np.float64)
        forward_v = float(velocity[0]) if velocity.size >= 1 else 0.0
        lateral_v = float(velocity[1]) if velocity.size >= 2 else 0.0
        limit = self.gains.search_max_velocity_brake_rate_rad_s

        # Internal convention: positive pitch-rate pitches up / brakes forward
        # flight; positive roll-rate rolls right, so it is opposite lateral
        # body velocity when damping drift.
        pitch_rate = np.clip(
            self.gains.search_forward_velocity_damping * forward_v,
            -limit,
            limit,
        )
        roll_rate = np.clip(
            -self.gains.search_lateral_velocity_damping * lateral_v,
            -limit,
            limit,
        )
        return float(roll_rate), float(pitch_rate)

    def _search_velocity_is_settled(self, telemetry: VehicleTelemetry) -> bool:
        body_velocity = telemetry.linear_velocity_m_s
        if body_velocity is None:
            return True

        velocity = np.asarray(body_velocity, dtype=np.float64)
        if velocity.size == 0:
            return True
        horizontal = velocity[: min(2, velocity.size)]
        return float(np.linalg.norm(horizontal)) <= self.gains.search_velocity_settle_m_s

    def _forward_pitch_rate(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry,
        confidence_scale: float,
        centered_scale: float,
    ) -> float:
        if gate.distance_m is None:
            approach = 0.35
        else:
            distance_error = gate.distance_m - self.gains.target_pass_distance_m
            approach = float(np.clip(distance_error / self.gains.far_distance_m, 0.0, 1.0))

        speed_damping = 0.0
        if telemetry.linear_velocity_m_s is not None and len(telemetry.linear_velocity_m_s) >= 1:
            speed_damping = self.gains.damping_pitch_from_speed * max(
                0.0,
                float(telemetry.linear_velocity_m_s[0]),
            )

        # Negative pitch-rate means "push nose down / accelerate forward" for
        # the internal adapter convention. Wire-level adapters own sign checks.
        # Forward suppression uses an earth-relative vertical cue when attitude
        # is available. Thrust stays body/camera-relative so the first gate
        # still gets conservative climb authority while it is low in the FPV.
        vertical_control_rad = self._forward_vertical_bearing(gate, telemetry)
        vertical_error = max(
            0.0,
            abs(vertical_control_rad) - self.gains.vertical_forward_deadband_rad,
        )
        vertical_centered_scale = 1.0 - vertical_error / max(
            self.gains.vertical_forward_suppression_rad,
            1e-6,
        )
        vertical_centered_scale = float(np.clip(vertical_centered_scale, 0.0, 1.0))

        pitch_rate = (
            -self.gains.forward_gain
            * approach
            * confidence_scale
            * centered_scale
            * vertical_centered_scale
        )
        pitch_rate += speed_damping
        return pitch_rate

    def _body_vertical_bearing(self, gate: GateEstimate) -> float:
        """Convert camera-relative vertical bearing into body elevation.

        The FPV detector reports bearing relative to the tilted optical axis.
        The AGP spec camera points 20 degrees upward, so a gate can appear
        below image center while still being physically above the drone.
        """
        return float(gate.bearing_v_rad + self.gains.camera_tilt_up_rad)

    def _forward_vertical_bearing(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry,
    ) -> float:
        body_elevation = self._body_vertical_bearing(gate)
        if telemetry.rpy_rad is not None and len(telemetry.rpy_rad) >= 2:
            return float(body_elevation + telemetry.rpy_rad[1])
        if telemetry.attitude_quat is not None and len(telemetry.attitude_quat) >= 4:
            return float(
                body_elevation
                + body_forward_elevation_from_quat_xyzw(telemetry.attitude_quat)
            )
        return body_elevation

    def _body_forward_elevation(self, telemetry: VehicleTelemetry) -> float:
        if telemetry.rpy_rad is not None and len(telemetry.rpy_rad) >= 2:
            return float(telemetry.rpy_rad[1])
        if telemetry.attitude_quat is not None and len(telemetry.attitude_quat) >= 4:
            return body_forward_elevation_from_quat_xyzw(telemetry.attitude_quat)
        return 0.0

    @staticmethod
    def _timestamp_s(telemetry: VehicleTelemetry) -> float:
        if telemetry.timestamp_s > 0.0:
            return float(telemetry.timestamp_s)
        return time.monotonic()

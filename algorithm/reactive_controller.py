"""Reactive gate-following controller for completion-first VQ1 racing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry


@dataclass(frozen=True)
class ReactiveControllerGains:
    """Controller gains and safety limits."""

    yaw_gain: float = 2.2
    roll_gain: float = 0.35
    vertical_gain: float = 0.30
    forward_gain: float = 0.45
    damping_pitch_from_speed: float = 0.04
    hover_thrust: float = 0.52
    min_thrust: float = 0.30
    max_thrust: float = 0.78
    search_yaw_rate_rad_s: float = 0.45
    max_roll_rate_rad_s: float = 0.70
    max_pitch_rate_rad_s: float = 0.80
    max_yaw_rate_rad_s: float = 1.20
    target_pass_distance_m: float = 1.4
    far_distance_m: float = 5.5
    minimum_track_confidence: float = 0.05


class ReactiveGateController:
    """Map a tracked gate estimate into body-rate/thrust commands."""

    def __init__(self, gains: ReactiveControllerGains | None = None) -> None:
        self.gains = gains or ReactiveControllerGains()

    def compute(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry | None = None,
    ) -> RacingCommand:
        telemetry = telemetry or VehicleTelemetry()

        if not gate.is_usable or gate.confidence < self.gains.minimum_track_confidence:
            return RacingCommand(
                yaw_rate_rad_s=self.gains.search_yaw_rate_rad_s,
                thrust_norm=self.gains.hover_thrust,
                mode=TrackMode.SEARCH,
            ).clipped(
                self.gains.max_roll_rate_rad_s,
                self.gains.max_pitch_rate_rad_s,
                self.gains.max_yaw_rate_rad_s,
                self.gains.min_thrust,
                self.gains.max_thrust,
            )

        confidence_scale = float(np.clip(gate.confidence, 0.15, 1.0))
        centered_scale = float(np.clip(1.0 - abs(gate.bearing_h_rad) / 0.9, 0.25, 1.0))

        yaw_rate = self.gains.yaw_gain * gate.bearing_h_rad * confidence_scale
        roll_rate = self.gains.roll_gain * gate.bearing_h_rad * confidence_scale
        thrust = self.gains.hover_thrust + self.gains.vertical_gain * gate.bearing_v_rad

        pitch_rate = self._forward_pitch_rate(gate, telemetry, confidence_scale, centered_scale)

        if gate.mode == TrackMode.COMMIT:
            pitch_rate *= 1.15
            yaw_rate *= 0.75

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
        pitch_rate = -self.gains.forward_gain * approach * confidence_scale * centered_scale
        pitch_rate += speed_damping
        return pitch_rate

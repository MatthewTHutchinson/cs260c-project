"""End-to-end VQ1 autonomous racing pilot."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from algorithm.detector_factory import build_gate_tracker
from algorithm.gate_tracker import GateTracker
from algorithm.reactive_controller import ReactiveGateController
from algorithm.types import GateEstimate, RacingCommand, VehicleTelemetry

if TYPE_CHECKING:
    from algorithm.learned_controller import LearnedFeatureController


class AutonomousRacingPilot:
    """Perception-guided body-rate autopilot.

    Inputs:
    - one FPV frame, when available
    - allowed telemetry such as attitude, angular rates, linear velocity, IMU

    Output:
    - `RacingCommand(roll_rate, pitch_rate, yaw_rate, thrust)`
    """

    def __init__(
        self,
        tracker: Optional[GateTracker] = None,
        controller: Optional[ReactiveGateController] = None,
        learned_controller: Optional[LearnedFeatureController] = None,
        frame_format: str = "bgr",
        learned_max_abs_bearing_h_rad: float = 0.45,
        learned_max_abs_bearing_v_rad: float = 0.50,
        learned_max_distance_m: float = 8.0,
        learned_min_confidence: float = 0.35,
        learned_max_abs_lateral_velocity_m_s: float = 2.0,
    ) -> None:
        self.tracker = tracker or build_gate_tracker()
        self.controller = controller or ReactiveGateController()
        self.learned_controller = learned_controller
        self.frame_format = frame_format
        self.learned_max_abs_bearing_h_rad = float(learned_max_abs_bearing_h_rad)
        self.learned_max_abs_bearing_v_rad = float(learned_max_abs_bearing_v_rad)
        self.learned_max_distance_m = float(learned_max_distance_m)
        self.learned_min_confidence = float(learned_min_confidence)
        self.learned_max_abs_lateral_velocity_m_s = float(
            learned_max_abs_lateral_velocity_m_s
        )
        self.last_command_source = "reactive"

    def reset(self) -> None:
        self.tracker.reset()
        self.controller.reset()
        if self.learned_controller is not None:
            self.learned_controller.reset()
        self.last_command_source = "reactive"

    def update(
        self,
        frame: Optional[np.ndarray],
        telemetry: Optional[VehicleTelemetry] = None,
        timestamp_s: Optional[float] = None,
    ) -> tuple[RacingCommand, GateEstimate]:
        telemetry = telemetry or VehicleTelemetry()
        timestamp = float(
            timestamp_s
            if timestamp_s is not None
            else telemetry.timestamp_s
            if telemetry.timestamp_s > 0.0
            else time.monotonic()
        )

        gate = self.tracker.update(
            frame,
            timestamp_s=timestamp,
            frame_format=self.frame_format,
        )
        if self.learned_controller is not None and self._learned_is_safe(
            gate, telemetry
        ):
            command = self.learned_controller.compute(gate, telemetry)
            self.last_command_source = "learned"
        else:
            command = self.controller.compute(gate, telemetry)
            self.last_command_source = (
                "reactive_fallback"
                if self.learned_controller is not None
                else "reactive"
            )
        return command, gate

    def _learned_is_safe(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry,
    ) -> bool:
        """Use learned control only inside the current feature-policy envelope."""
        if not gate.is_usable:
            return False
        if gate.confidence < self.learned_min_confidence:
            return False
        if abs(gate.bearing_h_rad) > self.learned_max_abs_bearing_h_rad:
            return False
        if abs(gate.bearing_v_rad) > self.learned_max_abs_bearing_v_rad:
            return False
        if gate.distance_m is not None and gate.distance_m > self.learned_max_distance_m:
            return False
        velocity = telemetry.linear_velocity_m_s
        if velocity is not None and len(velocity) >= 2:
            if abs(float(velocity[1])) > self.learned_max_abs_lateral_velocity_m_s:
                return False
        return True

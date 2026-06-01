"""End-to-end VQ1 autonomous racing pilot."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from algorithm.gate_tracker import GateTracker
from algorithm.reactive_controller import ReactiveGateController
from algorithm.types import GateEstimate, RacingCommand, VehicleTelemetry


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
        frame_format: str = "bgr",
    ) -> None:
        self.tracker = tracker or GateTracker()
        self.controller = controller or ReactiveGateController()
        self.frame_format = frame_format

    def reset(self) -> None:
        self.tracker.reset()

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
        command = self.controller.compute(gate, telemetry)
        return command, gate

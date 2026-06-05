"""Runtime wrapper for feature-policy checkpoints.

This keeps the learned policy behind the same `RacingCommand` boundary as the
reactive controller. It consumes only FPV-derived gate estimates, short history,
and allowed telemetry.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry
from learning.datasets import FeatureSpec, MODE_NAMES, TARGET_COLUMNS
from learning.feature_policy import load_checkpoint


class LearnedFeatureController:
    """GRU feature-policy inference wrapper with command clipping."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        max_roll_rate_rad_s: float = 0.70,
        max_pitch_rate_rad_s: float = 0.80,
        max_yaw_rate_rad_s: float = 1.20,
        min_thrust_norm: float = 0.30,
        max_thrust_norm: float = 0.95,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.payload = load_checkpoint(str(checkpoint), map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.feature_names = tuple(self.payload.get("feature_names", FeatureSpec.default().feature_names))
        self.sequence_length = int(self.payload.get("metadata", {}).get("sequence_length", 12))
        self.mean = self.payload["feature_mean"].to(self.device).float()
        self.std = self.payload["feature_std"].to(self.device).float()
        self.active_features = self.std >= 1e-6
        self.safe_std = torch.where(self.active_features, self.std, torch.ones_like(self.std))
        self.max_roll_rate_rad_s = float(max_roll_rate_rad_s)
        self.max_pitch_rate_rad_s = float(max_pitch_rate_rad_s)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self.min_thrust_norm = float(min_thrust_norm)
        self.max_thrust_norm = float(max_thrust_norm)
        self._history: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._prev_command = RacingCommand()
        self._prev_bearing_h = 0.0
        self._prev_bearing_v = 0.0
        self._prev_distance = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._prev_command = RacingCommand()
        self._prev_bearing_h = 0.0
        self._prev_bearing_v = 0.0
        self._prev_distance = 0.0

    def compute(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry | None = None,
    ) -> RacingCommand:
        telemetry = telemetry or VehicleTelemetry()
        feature = self._feature_vector(gate, telemetry)
        self._history.append(feature)
        while len(self._history) < self.sequence_length:
            self._history.appendleft(feature.copy())

        x = torch.from_numpy(np.stack(tuple(self._history)).astype(np.float32))
        x = x.unsqueeze(0).to(self.device)
        x = (x - self.mean) / self.safe_std
        x = torch.where(self.active_features, x, torch.zeros_like(x))
        with torch.no_grad():
            y = self.model(x).squeeze(0).detach().cpu().numpy()

        command = RacingCommand(
            roll_rate_rad_s=float(y[0]),
            pitch_rate_rad_s=float(y[1]),
            yaw_rate_rad_s=float(y[2]),
            thrust_norm=float(y[3]),
            mode=gate.mode,
        ).clipped(
            self.max_roll_rate_rad_s,
            self.max_pitch_rate_rad_s,
            self.max_yaw_rate_rad_s,
            self.min_thrust_norm,
            self.max_thrust_norm,
        )
        self._prev_command = command
        self._prev_bearing_h = float(gate.bearing_h_rad)
        self._prev_bearing_v = float(gate.bearing_v_rad)
        self._prev_distance = float(gate.distance_m or 0.0)
        return command

    def _feature_vector(
        self,
        gate: GateEstimate,
        telemetry: VehicleTelemetry,
    ) -> np.ndarray:
        distance = float(gate.distance_m or 0.0)
        pixel_x = float(gate.pixel_center[0]) if gate.pixel_center else 0.0
        pixel_y = float(gate.pixel_center[1]) if gate.pixel_center else 0.0
        velocity = telemetry.linear_velocity_m_s
        values = {
            "frame_fresh": 1.0 if gate.mode in {TrackMode.DETECTED, TrackMode.COMMIT} else 0.0,
            "last_gate_passed": float(max(-1, gate.sequence_index - 1)),
            "next_gate_index": float(gate.sequence_index),
            "body_forward_elevation_rad": self._body_forward_elevation(telemetry),
            "body_vx_m_s": self._velocity_component(velocity, 0),
            "body_vy_m_s": self._velocity_component(velocity, 1),
            "body_vz_m_s": self._velocity_component(velocity, 2),
            "confidence": float(gate.confidence),
            "bearing_h_rad": float(gate.bearing_h_rad),
            "bearing_v_rad": float(gate.bearing_v_rad),
            "distance_m": distance,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "apparent_size_px": float(gate.apparent_size_px or 0.0),
            "gate_age_s": float(gate.age_s),
            "has_distance": 1.0 if distance > 0.0 else 0.0,
            "bearing_h_delta": float(gate.bearing_h_rad) - self._prev_bearing_h,
            "bearing_v_delta": float(gate.bearing_v_rad) - self._prev_bearing_v,
            "distance_delta": distance - self._prev_distance,
        }
        mode = gate.mode.value if isinstance(gate.mode, TrackMode) else str(gate.mode)
        for mode_name in MODE_NAMES:
            values[f"mode_{mode_name}"] = 1.0 if mode == mode_name else 0.0
        for name in TARGET_COLUMNS:
            values[f"prev_{name}"] = self._prev_command_value(name)

        return np.asarray([values.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    @staticmethod
    def _velocity_component(velocity: np.ndarray | None, index: int) -> float:
        if velocity is None or len(velocity) <= index:
            return 0.0
        return float(velocity[index])

    @staticmethod
    def _body_forward_elevation(telemetry: VehicleTelemetry) -> float:
        if telemetry.rpy_rad is not None and len(telemetry.rpy_rad) >= 2:
            return float(telemetry.rpy_rad[1])
        return 0.0

    def _prev_command_value(self, name: str) -> float:
        if name == "roll_rate_rad_s":
            return float(self._prev_command.roll_rate_rad_s)
        if name == "pitch_rate_rad_s":
            return float(self._prev_command.pitch_rate_rad_s)
        if name == "yaw_rate_rad_s":
            return float(self._prev_command.yaw_rate_rad_s)
        if name == "thrust_norm":
            return float(self._prev_command.thrust_norm)
        return 0.0

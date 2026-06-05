"""Runtime wrapper for feature-policy checkpoints.

This keeps the learned policy behind the same `RacingCommand` boundary as the
reactive controller. It consumes only FPV-derived gate estimates, short history,
and allowed telemetry.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry


MODE_NAMES = ("search", "tracked", "detected", "commit", "recover")
TARGET_COLUMNS = (
    "roll_rate_rad_s",
    "pitch_rate_rad_s",
    "yaw_rate_rad_s",
    "thrust_norm",
)
DEFAULT_FEATURE_NAMES = (
    "frame_fresh",
    "last_gate_passed",
    "next_gate_index",
    "body_forward_elevation_rad",
    "body_vx_m_s",
    "body_vy_m_s",
    "body_vz_m_s",
    "confidence",
    "bearing_h_rad",
    "bearing_v_rad",
    "distance_m",
    "pixel_x",
    "pixel_y",
    "apparent_size_px",
    "gate_age_s",
    *[f"mode_{mode}" for mode in MODE_NAMES],
    *[f"prev_{name}" for name in TARGET_COLUMNS],
    "bearing_h_delta",
    "bearing_v_delta",
    "distance_delta",
    "has_distance",
)


class LearnedFeatureController:
    """GRU feature-policy inference wrapper with command clipping."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        max_roll_rate_rad_s: float = 0.70,
        max_pitch_rate_rad_s: float = 0.80,
        max_yaw_rate_rad_s: float = 1.20,
        min_thrust_norm: float = 0.30,
        max_thrust_norm: float = 0.95,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = str(device)
        if self.checkpoint.suffix == ".npz":
            self._load_npz(self.checkpoint)
        else:
            self._load_torch(self.checkpoint)
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

    def _load_npz(self, checkpoint: Path) -> None:
        payload = np.load(checkpoint, allow_pickle=False)
        self.runtime = "numpy"
        self.feature_names = tuple(str(x) for x in payload["feature_names"].tolist())
        self.sequence_length = int(payload["sequence_length"])
        self.mean = payload["feature_mean"].astype(np.float32)
        self.std = payload["feature_std"].astype(np.float32)
        self.active_features = self.std >= 1e-6
        self.safe_std = np.where(self.active_features, self.std, 1.0).astype(np.float32)
        self.gru_weight_ih = payload["gru_weight_ih"].astype(np.float32)
        self.gru_weight_hh = payload["gru_weight_hh"].astype(np.float32)
        self.gru_bias_ih = payload["gru_bias_ih"].astype(np.float32)
        self.gru_bias_hh = payload["gru_bias_hh"].astype(np.float32)
        self.ln_weight = payload["ln_weight"].astype(np.float32)
        self.ln_bias = payload["ln_bias"].astype(np.float32)
        self.head1_weight = payload["head1_weight"].astype(np.float32)
        self.head1_bias = payload["head1_bias"].astype(np.float32)
        self.head2_weight = payload["head2_weight"].astype(np.float32)
        self.head2_bias = payload["head2_bias"].astype(np.float32)

    def _load_torch(self, checkpoint: Path) -> None:
        import torch
        from learning.feature_policy import load_checkpoint

        self.torch = torch
        self.torch_device = torch.device(self.device)
        self.model, self.payload = load_checkpoint(str(checkpoint), map_location=self.torch_device)
        self.model.to(self.torch_device)
        self.model.eval()
        self.runtime = "torch"
        self.feature_names = tuple(self.payload.get("feature_names", DEFAULT_FEATURE_NAMES))
        self.sequence_length = int(self.payload.get("metadata", {}).get("sequence_length", 12))
        self.mean = self.payload["feature_mean"].to(self.torch_device).float()
        self.std = self.payload["feature_std"].to(self.torch_device).float()
        self.active_features = self.std >= 1e-6
        self.safe_std = torch.where(self.active_features, self.std, torch.ones_like(self.std))

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

        x_np = np.stack(tuple(self._history)).astype(np.float32)
        if self.runtime == "numpy":
            x_np = (x_np - self.mean) / self.safe_std
            x_np = np.where(self.active_features, x_np, 0.0).astype(np.float32)
            y = self._numpy_forward(x_np)
        else:
            torch = self.torch
            x = torch.from_numpy(x_np).unsqueeze(0).to(self.torch_device)
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

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        hidden_dim = self.gru_weight_hh.shape[1]
        h = np.zeros(hidden_dim, dtype=np.float32)
        w_ir, w_iz, w_in = np.split(self.gru_weight_ih, 3, axis=0)
        w_hr, w_hz, w_hn = np.split(self.gru_weight_hh, 3, axis=0)
        b_ir, b_iz, b_in = np.split(self.gru_bias_ih, 3)
        b_hr, b_hz, b_hn = np.split(self.gru_bias_hh, 3)
        for xt in x:
            r = self._sigmoid(w_ir @ xt + b_ir + w_hr @ h + b_hr)
            z = self._sigmoid(w_iz @ xt + b_iz + w_hz @ h + b_hz)
            n = np.tanh(w_in @ xt + b_in + r * (w_hn @ h + b_hn))
            h = (1.0 - z) * n + z * h

        h = self._layer_norm(h)
        h = self._silu(self.head1_weight @ h + self.head1_bias)
        return self.head2_weight @ h + self.head2_bias

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x)
        var = np.mean((x - mean) ** 2)
        return (x - mean) / np.sqrt(var + eps) * self.ln_weight + self.ln_bias

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

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

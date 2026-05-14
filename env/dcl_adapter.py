"""DCL Competition Platform Adapter.

This file is a scaffold, not a working competition client.
It has not yet been wired to the official MAVLink / UDP runtime described in
`docs/260508_Technical_Spec_0002.pdf`, and it should fail fast if used.

Architecture
------------
The adapter implements AbstractDroneRacingEnv so the trained policy and
expert can run unchanged against the competition platform.

Observation building
--------------------
The latest spec provides telemetry plus a forward camera stream.
This scaffold still sketches only a legacy 12-D bridge
(velocity, angular rate, next-two-gate geometry), which is not compatible
with the repo's current 78-D richer-state models or the multimodal image
policies without additional work.

Control
-------
The latest spec explicitly lists MAVLink messages such as
`SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`.
Older project notes talked about Throttle/Roll/Pitch/Yaw commands, but that
should now be treated as historical wording rather than a confirmed
competition interface.
"""

from collections import deque
from typing import Optional

import numpy as np

from env.abstract_env import AbstractDroneRacingEnv
from env.gate_detector import GateDetector, GateObservation, CameraParams


# -----------------------------------------------------------------------
# Minimal RPYT PID (body-rate / altitude controller stub)
# -----------------------------------------------------------------------

class _WaypointToPIDController:
    """Converts body-frame waypoint delta → (throttle, roll, pitch, yaw) commands.

    This mirrors the logic in GateRaceAviary._preprocessAction but outputs
    normalised [-1, 1] RPYT instead of raw RPMs.

    TODO: Tune PID gains once DCL simulator physics are known.
    """

    def __init__(self, kp_xy=0.8, kp_z=1.2, kp_yaw=1.0):
        self.kp_xy  = kp_xy
        self.kp_z   = kp_z
        self.kp_yaw = kp_yaw

    def compute(
        self,
        pos:   np.ndarray,   # current world position
        vel:   np.ndarray,   # current world velocity
        rpy:   np.ndarray,   # current roll-pitch-yaw (radians)
        action: np.ndarray,  # normalised [-1,1] waypoint delta action
        clip_radius: float,
        max_dyaw:    float,
    ) -> np.ndarray:
        """Returns [throttle, roll, pitch, yaw_rate] each in [-1, 1]."""
        from scipy.spatial.transform import Rotation
        R_wb = Rotation.from_euler("xyz", rpy).as_matrix()  # body→world

        delta_body = action[:3] * clip_radius
        delta_yaw  = action[3]  * max_dyaw

        target_world = pos + R_wb @ delta_body
        err_world    = target_world - pos

        # Map world error to roll/pitch commands (simplified)
        yaw = rpy[2]
        err_fwd  =  err_world[0] * np.cos(yaw) + err_world[1] * np.sin(yaw)
        err_side = -err_world[0] * np.sin(yaw) + err_world[1] * np.cos(yaw)
        err_z    =  err_world[2]

        pitch    = float(np.clip(-self.kp_xy * err_fwd,  -1.0, 1.0))
        roll     = float(np.clip( self.kp_xy * err_side, -1.0, 1.0))
        throttle = float(np.clip(0.5 + self.kp_z * err_z, 0.0, 1.0))
        yaw_cmd  = float(np.clip(self.kp_yaw * delta_yaw, -1.0, 1.0))

        return np.array([throttle, roll, pitch, yaw_cmd], dtype=np.float32)


# -----------------------------------------------------------------------
# DCL Adapter
# -----------------------------------------------------------------------

class DCLRacingEnv(AbstractDroneRacingEnv):
    """Wraps the future DCL competition API behind AbstractDroneRacingEnv.

    Important:
    This class is currently a non-operational scaffold. It is preserved to
    document the intended abstraction boundary, but it does not yet implement
    the real MAVLink control loop, UDP vision stream reassembly, or the
    current richer observation formats used by the repo's best policies.

    Parameters
    ----------
    dcl_api : object
        DCL platform API handle.  TODO: type and constructor once docs arrive.
    gate_detector : GateDetector
        Vision-based gate detector tuned for the current qualifier round.
    camera_params : CameraParams
        Intrinsics of the drone's forward camera.
    cam_to_body_R : (3,3) ndarray, optional
        Rotation from camera frame to body frame.
    history_len : int
        Must match the value used during training.
    clip_radius : float
        Action scale — must match the trained policy's value.
    max_dyaw : float
        Yaw action scale — must match the trained policy's value.
    n_gates : int
        Number of gates in the current course layout.
    """

    def __init__(
        self,
        dcl_api,
        gate_detector: GateDetector,
        camera_params: CameraParams = None,
        cam_to_body_R: Optional[np.ndarray] = None,
        history_len: int = 3,
        clip_radius: float = 1.0,
        max_dyaw: float = 0.3,
        n_gates: int = 10,
    ):
        self._api          = dcl_api
        self._detector     = gate_detector
        self._cam          = camera_params or CameraParams()
        self._cam_to_body  = cam_to_body_R
        self.history_len   = history_len
        self.clip_radius   = clip_radius
        self.max_dyaw      = max_dyaw
        self._n_gates      = n_gates
        self._pid          = _WaypointToPIDController()

        self._obs_buffer   = deque(maxlen=history_len)
        self._next_gate    = 0
        self._gates_passed = 0
        self._last_gate_positions: list[np.ndarray] = []

        import gymnasium as gym
        obs_dim = 12 * history_len
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high= np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high= np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # AbstractDroneRacingEnv interface
    # ------------------------------------------------------------------

    @property
    def n_gates_total(self) -> int:
        return self._n_gates

    def set_clip_radius(self, r: float) -> None:
        self.clip_radius = float(r)

    def get_full_state(self) -> dict:
        telemetry = self._get_telemetry()
        return {
            "pos":         telemetry["pos"],
            "quat":        telemetry["quat"],
            "rpy":         telemetry["rpy"],
            "vel":         telemetry["vel"],
            "ang_vel":     telemetry["ang_vel"],
            "next_gate":   self._next_gate,
            "clip_radius": self.clip_radius,
            "max_dyaw":    self.max_dyaw,
        }

    def reset(self, seed=None, options=None):
        raise NotImplementedError(
            "DCLRacingEnv is a scaffold only. "
            "It has not yet been wired to the official MAVLink / UDP runtime."
        )

    def step(self, action: np.ndarray):
        raise NotImplementedError(
            "DCLRacingEnv is a scaffold only. "
            "Implement MAVLink control, telemetry polling, and vision reassembly first."
        )

    def close(self) -> None:
        # TODO: self._api.close()
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_telemetry(self) -> dict:
        """Poll the DCL API for current drone state."""
        raise NotImplementedError("Telemetry polling is not implemented for DCLRacingEnv yet.")

    def _get_frame(self) -> Optional[np.ndarray]:
        """Get the latest decoded camera frame from the DCL API."""
        raise NotImplementedError("Vision-stream decoding is not implemented for DCLRacingEnv yet.")

    def _build_obs(self) -> np.ndarray:
        """Construct the history-stacked 12-D observation for the policy."""
        telemetry = self._get_telemetry()
        pos     = telemetry["pos"]
        quat    = telemetry["quat"]
        vel_w   = telemetry["vel"]
        ang_w   = telemetry["ang_vel"]

        from scipy.spatial.transform import Rotation
        rot_T = Rotation.from_quat(quat).as_matrix().T  # world→body

        vel_body = rot_T @ vel_w
        ang_body = rot_T @ ang_w

        # Try to estimate gate positions from camera
        frame = self._get_frame()
        rel_g0, rel_g1 = self._estimate_gate_positions(frame, pos, rot_T)

        single = np.concatenate([vel_body, ang_body, rel_g0, rel_g1]).astype(np.float32)
        self._obs_buffer.append(single)
        return np.concatenate(list(self._obs_buffer)).astype(np.float32)

    def _estimate_gate_positions(
        self,
        frame: Optional[np.ndarray],
        pos: np.ndarray,
        rot_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate relative gate positions in body frame.

        Uses camera detection when a frame is available; falls back to
        last known positions when detection fails (e.g. gate out of FOV).
        """
        default = np.zeros(3, dtype=np.float32)

        if frame is None:
            # No frame — use cached positions if available
            if len(self._last_gate_positions) >= 2:
                g0 = rot_T @ (self._last_gate_positions[0] - pos)
                g1 = rot_T @ (self._last_gate_positions[1] - pos)
                return g0.astype(np.float32), g1.astype(np.float32)
            return default.copy(), default.copy()

        detected = self._detector.detect(frame, self._cam, max_gates=2)

        positions = []
        for gate_obs in detected:
            rel_cam = self._detector.gate_obs_to_body_frame(gate_obs, self._cam_to_body)
            rel_world = rot_T.T @ rel_cam  # body → world offset
            gate_world = pos + rel_world
            positions.append(gate_world)

        # Cache for fallback
        self._last_gate_positions = positions

        rel_g0 = (rot_T @ (positions[0] - pos)).astype(np.float32) if len(positions) > 0 else default.copy()
        rel_g1 = (rot_T @ (positions[1] - pos)).astype(np.float32) if len(positions) > 1 else default.copy()
        return rel_g0, rel_g1

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from env.gate_detector import CameraParams, GateDetector
from env.tracks import get_track, get_tracks


# Rectangular CCW circuit (viewed from above).
# Gate normals point in the direction of correct traversal.
DEFAULT_GATES = [
    dict(center=np.array([2.5, 0.0, 1.5]), normal=np.array([1.0, 0.0, 0.0]), radius=0.5),
    dict(center=np.array([5.0, 2.5, 1.5]), normal=np.array([0.0, 1.0, 0.0]), radius=0.5),
    dict(center=np.array([2.5, 5.0, 1.5]), normal=np.array([-1.0, 0.0, 0.0]), radius=0.5),
    dict(center=np.array([0.0, 2.5, 1.5]), normal=np.array([0.0, -1.0, 0.0]), radius=0.5),
]

_INIT_XYZ = np.array([[-0.5, 0.0, 1.5]])
_INIT_RPY = np.array([[0.0, 0.0, 0.0]])

_TEXTURE_DIR = Path(__file__).resolve().parent.parent / "assets" / "textures"
_FLOOR_TEXTURES = ["floor_checker.png", "floor_concrete.png", "floor_rubber.png"]
_WALL_TEXTURES = ["wall_panel.png", "wall_grid.png"]


def _normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def _horizontal_lateral(normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    lateral = np.array([-normal[1], normal[0], 0.0], dtype=np.float64)
    if np.linalg.norm(lateral[:2]) < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _normalize(lateral)


def _quat_from_rotation_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to a quaternion [x, y, z, w]."""
    m = np.asarray(rot, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-9)
    return tuple(quat.tolist())


def _quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> tuple[float, float, float, float]:
    rot = np.column_stack([_normalize(x_axis), _normalize(y_axis), _normalize(z_axis)])
    return _quat_from_rotation_matrix(rot)


def compute_single_obs_dim(
    lookahead_gates: int = 2,
    include_gate_normals: bool = False,
    include_relative_heading: bool = False,
) -> int:
    """Per-timestep observation dimension for the configured feature set."""
    dim = 6 + 3 * int(lookahead_gates)  # vel_body + ang_body + gate positions
    if include_gate_normals:
        dim += 3 * int(lookahead_gates)
    if include_relative_heading:
        dim += 2  # cos/sin of heading error to the next gate normal
    return dim


def compute_obs_dim_from_config(cfg: dict) -> int:
    """Total stacked observation dimension from an env config dict."""
    history_len = int(cfg.get("history_len", 3))
    return history_len * compute_single_obs_dim(
        lookahead_gates=int(cfg.get("lookahead_gates", 2)),
        include_gate_normals=bool(cfg.get("include_gate_normals", False)),
        include_relative_heading=bool(cfg.get("include_relative_heading", False)),
    )


class GateRaceAviary(BaseAviary):
    """Single-drone gate racing environment with optional onboard vision.

    Default observations remain vector-only and backward compatible.
    When `observation_source="vision_bridge"`, gate-relative positions are
    reconstructed from the onboard RGB camera using GateDetector, while
    dynamics terms (body velocity, body angular rate) still come from the
    simulator state. This gives a low-risk perception bridge without
    rewriting the existing policy interfaces.
    """

    def __init__(
        self,
        gates=None,
        track_options=None,
        track_names: list[str] | None = None,
        sample_track_on_reset: bool = False,
        ctrl_freq: int = 20,
        pyb_freq: int = 240,
        clip_radius: float = 0.25,
        max_dyaw: float = 0.3,
        episode_len_sec: float = 12.0,
        history_len: int = 3,
        lookahead_gates: int = 2,
        include_gate_normals: bool = False,
        include_relative_heading: bool = False,
        smooth_coef: float = 0.01,
        progress_coef: float = 0.1,
        gui: bool = False,
        record: bool = False,
        camera_follow: bool = False,
        camera_distance: float = 3.2,
        camera_yaw: float = -35.0,
        camera_pitch: float = -28.0,
        start_distance: float = 3.0,
        start_longitudinal_jitter: float = 0.0,
        start_lateral_jitter: float = 0.0,
        start_vertical_jitter: float = 0.0,
        start_yaw_jitter: float = 0.0,
        disturbance_force_std: float = 0.0,
        disturbance_torque_std: float = 0.0,
        observation_source: str = "state",
        vision_bridge_fallback: str = "cache_track",
        vision_bridge_normal_source: str = "track",
        vision_bridge_min_confidence: float = 0.1,
        enable_onboard_camera: bool = False,
        camera_width: int = 128,
        camera_height: int = 96,
        camera_fov: float = 85.0,
        camera_tilt_deg: float = 8.0,
        camera_offset_forward: float = 0.08,
        camera_offset_up: float = 0.03,
        camera_near: float = 0.03,
        camera_far: float = 25.0,
        camera_noise_std: float = 0.0,
        camera_exposure_jitter: float = 0.0,
        camera_occlusion_prob: float = 0.0,
        camera_occlusion_count: int = 1,
        camera_include_in_info: bool = False,
        camera_shadow: bool = True,
        scene_visuals: bool = False,
        scene_randomization: bool = False,
        scene_clutter_count: int = 8,
    ):
        base_track = gates if gates is not None else DEFAULT_GATES
        self._track_options = track_options or [base_track]
        self._track_names = track_names or [f"track_{i}" for i in range(len(self._track_options))]
        self._sample_track_on_reset = bool(sample_track_on_reset and len(self._track_options) > 1)
        self._track_rng = np.random.default_rng()
        self._disturbance_rng = np.random.default_rng()
        self._visual_rng = np.random.default_rng()
        self._camera_rng = np.random.default_rng()
        self._current_track_idx = 0
        self.gates = self._clone_track(self._track_options[self._current_track_idx])
        self._validate_track_options()
        self.n_gates = len(self.gates)
        self.clip_radius = clip_radius
        self.max_dyaw = max_dyaw
        self._episode_len_sec = episode_len_sec
        self.history_len = history_len
        self.lookahead_gates = max(1, int(lookahead_gates))
        self.include_gate_normals = bool(include_gate_normals)
        self.include_relative_heading = bool(include_relative_heading)
        self.single_obs_dim = compute_single_obs_dim(
            lookahead_gates=self.lookahead_gates,
            include_gate_normals=self.include_gate_normals,
            include_relative_heading=self.include_relative_heading,
        )
        self.smooth_coef = smooth_coef
        self.progress_coef = progress_coef
        self._gui_enabled = gui
        self._camera_follow = camera_follow and gui
        self._camera_distance = camera_distance
        self._camera_yaw = camera_yaw
        self._camera_pitch = camera_pitch
        self._start_distance = start_distance
        self._start_longitudinal_jitter = start_longitudinal_jitter
        self._start_lateral_jitter = start_lateral_jitter
        self._start_vertical_jitter = start_vertical_jitter
        self._start_yaw_jitter = start_yaw_jitter
        self._disturbance_force_std = float(disturbance_force_std)
        self._disturbance_torque_std = float(disturbance_torque_std)
        self._debug_text_ids: list[int] = []

        self._observation_source = str(observation_source).strip().lower()
        if self._observation_source not in {"state", "vision_bridge"}:
            raise ValueError("observation_source must be 'state' or 'vision_bridge'.")
        self._vision_bridge_fallback = str(vision_bridge_fallback).strip().lower()
        self._vision_bridge_normal_source = str(vision_bridge_normal_source).strip().lower()
        self._vision_bridge_min_confidence = float(vision_bridge_min_confidence)

        self._camera_active = bool(enable_onboard_camera or self._observation_source == "vision_bridge")
        self._camera_info_in_info = bool(camera_include_in_info)
        self._camera_shadow = bool(camera_shadow)
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._camera_fov = float(camera_fov)
        self._camera_tilt_deg = float(camera_tilt_deg)
        self._camera_offset_forward = float(camera_offset_forward)
        self._camera_offset_up = float(camera_offset_up)
        self._camera_near = float(camera_near)
        self._camera_far = float(camera_far)
        self._camera_noise_std = float(camera_noise_std)
        self._camera_exposure_jitter = float(camera_exposure_jitter)
        self._camera_occlusion_prob = float(camera_occlusion_prob)
        self._camera_occlusion_count = max(0, int(camera_occlusion_count))
        self._camera_params = self._build_camera_params(
            width=self._camera_width,
            height=self._camera_height,
            fov_deg=self._camera_fov,
        )
        self._last_camera_frame: Optional[np.ndarray] = None
        self._last_camera_depth: Optional[np.ndarray] = None
        self._last_camera_detections = []
        self._last_detected_rel_gates: list[Optional[np.ndarray]] = [None] * self.lookahead_gates
        self._last_detection_confidences = np.zeros(self.lookahead_gates, dtype=np.float32)

        self._scene_visuals = bool(scene_visuals or self._camera_active or gui)
        self._scene_randomization = bool(scene_randomization)
        self._scene_clutter_count = max(0, int(scene_clutter_count))
        self._scene_body_ids: list[int] = []
        self._gate_visual_body_ids: list[list[int]] = []
        self._texture_ids: dict[str, int] = {}
        self._visual_theme = self._build_default_visual_theme()

        self._gate_detector: Optional[GateDetector] = None
        if self._observation_source == "vision_bridge":
            gate_width = self._estimate_gate_physical_width()
            self._gate_detector = GateDetector(
                gate_physical_width=gate_width,
                min_contour_area=max(40, int(0.002 * self._camera_width * self._camera_height)),
            )

        # Observation history buffer — must exist before super().__init__
        # because _observationSpace() is called from within super().__init__.
        self._obs_buffer: deque = deque(maxlen=history_len)
        self._fill_obs_buffer_zeros()

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=_INIT_XYZ.copy(),
            initial_rpys=_INIT_RPY.copy(),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=False,
            user_debug_gui=False,
        )

        self.pid = DSLPIDControl(drone_model=DroneModel.CF2X)
        self._episode_len_steps = int(episode_len_sec * ctrl_freq)

        # Episode state — reset in reset()
        self._next_gate: int = 0
        self._gates_passed: int = 0
        self._prev_gate_sides: np.ndarray = np.zeros(self.n_gates)
        self._prev_pos: np.ndarray = _INIT_XYZ[0].copy()
        self._last_gate_events: dict = {}
        self._last_action: np.ndarray | None = None
        self._current_action: np.ndarray = np.zeros(4)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def n_gates_total(self) -> int:
        return self.n_gates

    @property
    def camera_frame_shape(self) -> tuple[int, int, int]:
        return self._camera_height, self._camera_width, 3

    @property
    def policy_image_shape(self) -> tuple[int, int, int]:
        return 3, self._camera_height, self._camera_width

    def set_clip_radius(self, r: float) -> None:
        """Update clip_radius mid-training (curriculum)."""
        self.clip_radius = float(r)

    def get_full_state(self) -> dict:
        """Full world-frame state dict used by the expert and DCL adapter."""
        state = self._getDroneStateVector(0)
        return {
            "pos": state[0:3].copy(),
            "quat": state[3:7].copy(),
            "rpy": state[7:10].copy(),
            "vel": state[10:13].copy(),
            "ang_vel": state[13:16].copy(),
            "next_gate": self._next_gate,
            "clip_radius": self.clip_radius,
            "max_dyaw": self.max_dyaw,
            "track_name": self.track_name,
        }

    def get_last_camera_frame(self) -> Optional[np.ndarray]:
        if self._last_camera_frame is None:
            return None
        return self._last_camera_frame.copy()

    def render_onboard_camera(self) -> Optional[np.ndarray]:
        return self._render_onboard_camera()

    @property
    def track_name(self) -> str:
        return self._track_names[self._current_track_idx]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._track_rng = np.random.default_rng(seed)
            self._disturbance_rng = np.random.default_rng(seed + 17)
            self._visual_rng = np.random.default_rng(seed + 29)
            self._camera_rng = np.random.default_rng(seed + 43)
        self._select_track_for_reset()
        self.pid.reset()
        self._next_gate = 0
        self._gates_passed = 0
        self._last_gate_events = {}
        self._prev_pos = _INIT_XYZ[0].copy()
        self._last_action = None
        self._current_action = np.zeros(4)
        self._reset_camera_cache()
        self._clear_visual_handles()

        # Pre-fill history with zeros so the buffer is ready when
        # super().reset() calls _computeObs().
        self._fill_obs_buffer_zeros()
        _, _ = super().reset(seed=seed, options=options)
        self._randomize_start_pose()
        self._rebuild_scene_visuals()
        self._updateAndStoreKinematicInformation()
        self._fill_obs_buffer_zeros()
        obs = self._computeObs()
        info = self._computeInfo()
        self._prev_gate_sides = self._get_gate_sides()
        self._prev_pos = self.pos[0].copy()
        if self._gui_enabled:
            self._setup_debug_visuals()
            self._update_debug_visuals()
        info["track_name"] = self.track_name
        return obs, info

    def step(self, action):
        # Snapshot before physics for gate crossing and smoothness.
        self._prev_pos = self.pos[0].copy()
        self._prev_gate_sides = self._get_gate_sides()
        self._last_action = self._current_action.copy()
        self._apply_random_disturbance()
        obs, reward, terminated, truncated, info = super().step(action)
        if self._gui_enabled:
            self._update_debug_visuals()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # BaseAviary abstract methods
    # ------------------------------------------------------------------

    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

    def _observationSpace(self):
        dim = self.single_obs_dim * self.history_len
        return spaces.Box(
            low=-np.inf * np.ones(dim, dtype=np.float32),
            high=np.inf * np.ones(dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()[:4]
        self._current_action = action.astype(np.float32)  # store for smoothness

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        ang_vel = state[13:16]
        yaw = float(state[9])

        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)  # body→world

        delta_pos_body = action[:3] * self.clip_radius
        delta_yaw = float(action[3]) * self.max_dyaw

        target_pos = pos + rot @ delta_pos_body
        target_rpy = np.array([0.0, 0.0, yaw + delta_yaw])

        rpm, _, _ = self.pid.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
        )
        return rpm.reshape(1, 4)

    def _computeObs(self):
        single = self._compute_single_obs()
        self._obs_buffer.append(single)
        return np.concatenate(list(self._obs_buffer)).astype(np.float32)

    def _computeReward(self):
        curr_sides = self._get_gate_sides()
        events = self._detect_gate_crossing(self._prev_gate_sides, curr_sides)
        self._last_gate_events = events

        reward = 0.0

        # --- Sparse event rewards (dominant) ---
        if events["gate_passed"]:
            reward += 10.0
            self._gates_passed += 1
            self._next_gate = (self._next_gate + 1) % self.n_gates
        if events["wrong_way"]:
            reward -= 5.0
        if events["collision"]:
            reward -= 50.0

        # --- Dense shaping: velocity alignment toward next gate ---
        # Dot-product projection: reward for flying *toward* gate, not just proximity.
        # This is strictly better than Euclidean distance shaping because it cannot
        # be gamed by oscillating near the gate.
        pos = self.pos[0]
        vel = self.vel[0]  # world frame
        gate = self.gates[self._next_gate]["center"]
        gate_dir = gate - pos
        gate_dist = np.linalg.norm(gate_dir)
        if gate_dist > 1e-4:
            progress = np.dot(vel, gate_dir / gate_dist)
            reward += self.progress_coef * progress

        # --- Smoothness penalty: penalise action jerk ---
        if self._last_action is not None:
            jerk = np.linalg.norm(self._current_action - self._last_action)
            reward -= self.smooth_coef * jerk

        return float(reward)

    def _computeTerminated(self):
        if self._last_gate_events.get("collision", False):
            return True
        pos = self.pos[0]
        if pos[2] < 0.15 or pos[2] > 6.0:
            return True
        if np.abs(pos[0]) > 12.0 or np.abs(pos[1]) > 12.0:
            return True
        return False

    def _computeTruncated(self):
        return (self.step_counter // self.PYB_STEPS_PER_CTRL) >= self._episode_len_steps

    def _computeInfo(self):
        info = dict(self._last_gate_events)
        info["gates_passed"] = self._gates_passed
        info["next_gate"] = self._next_gate
        info["clip_radius"] = self.clip_radius
        info["track_name"] = self.track_name
        info["observation_source"] = self._observation_source
        if self._camera_active:
            info["camera_frame_shape"] = self.camera_frame_shape
            info["camera_detections"] = len(self._last_camera_detections)
            if self._camera_info_in_info and self._last_camera_frame is not None:
                info["camera_rgb"] = self._last_camera_frame.copy()
        return info

    # ------------------------------------------------------------------
    # Internal observation helpers
    # ------------------------------------------------------------------

    def _compute_single_obs(self) -> np.ndarray:
        """Configurable single-timestep observation."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel_world = state[10:13]
        ang_world = state[13:16]

        rot_T = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3).T  # world→body

        vel_body = rot_T @ vel_world
        ang_body = rot_T @ ang_world

        frame = None
        if self._camera_active:
            frame = self._render_onboard_camera()

        lookahead_gates = [
            self.gates[(self._next_gate + offset) % self.n_gates]
            for offset in range(self.lookahead_gates)
        ]

        if self._observation_source == "vision_bridge":
            rel_gate_positions = self._estimate_gate_positions_from_camera(pos, rot_T, frame)
        else:
            rel_gate_positions = [rot_T @ (gate["center"] - pos) for gate in lookahead_gates]

        obs_parts = [vel_body, ang_body]
        obs_parts.extend(rel_gate_positions)

        if self.include_gate_normals:
            for offset, gate in enumerate(lookahead_gates):
                obs_parts.append(self._gate_normal_feature(gate, rot_T, offset))

        if self.include_relative_heading:
            heading_features = self._relative_heading_features(lookahead_gates[0], rot_T)
            obs_parts.append(heading_features)

        return np.concatenate(obs_parts).astype(np.float32)

    def _gate_normal_feature(self, gate: dict, rot_T: np.ndarray, offset: int) -> np.ndarray:
        if self._observation_source != "vision_bridge":
            return rot_T @ gate["normal"]
        if self._vision_bridge_normal_source == "zeros":
            return np.zeros(3, dtype=np.float32)
        if self._vision_bridge_normal_source != "track":
            raise ValueError("vision_bridge_normal_source must be 'track' or 'zeros'.")
        return (rot_T @ gate["normal"]).astype(np.float32)

    def _relative_heading_features(self, next_gate: dict, rot_T: np.ndarray) -> np.ndarray:
        if self._observation_source == "vision_bridge" and self._vision_bridge_normal_source == "zeros":
            return np.array([1.0, 0.0], dtype=np.float32)

        next_gate_normal_body = rot_T @ next_gate["normal"]
        heading_xy = next_gate_normal_body[:2]
        heading_norm = np.linalg.norm(heading_xy)
        if heading_norm < 1e-9:
            return np.array([1.0, 0.0], dtype=np.float32)
        heading_xy = heading_xy / heading_norm
        return np.array([heading_xy[0], heading_xy[1]], dtype=np.float32)

    def _estimate_gate_positions_from_camera(
        self,
        pos: np.ndarray,
        rot_T: np.ndarray,
        frame: Optional[np.ndarray],
    ) -> list[np.ndarray]:
        if self._gate_detector is None or frame is None:
            return [self._fallback_gate_position(offset, pos, rot_T) for offset in range(self.lookahead_gates)]

        detections = self._gate_detector.detect(
            frame[:, :, ::-1],
            cam=self._camera_params,
            max_gates=self.lookahead_gates,
        )
        self._last_camera_detections = detections

        rel_positions: list[np.ndarray] = []
        for offset in range(self.lookahead_gates):
            if offset < len(detections) and detections[offset].confidence >= self._vision_bridge_min_confidence:
                rel_pos = self._gate_detector.gate_obs_to_body_frame(detections[offset]).astype(np.float32)
                self._last_detected_rel_gates[offset] = rel_pos.copy()
                self._last_detection_confidences[offset] = float(detections[offset].confidence)
            else:
                rel_pos = self._fallback_gate_position(offset, pos, rot_T)
                self._last_detection_confidences[offset] = 0.0
            rel_positions.append(rel_pos.astype(np.float32))
        return rel_positions

    def _fallback_gate_position(self, offset: int, pos: np.ndarray, rot_T: np.ndarray) -> np.ndarray:
        gate = self.gates[(self._next_gate + offset) % self.n_gates]
        track_rel = (rot_T @ (gate["center"] - pos)).astype(np.float32)
        cache_rel = self._last_detected_rel_gates[offset]
        zeros = np.zeros(3, dtype=np.float32)

        if self._vision_bridge_fallback == "track":
            return track_rel
        if self._vision_bridge_fallback == "cache":
            return cache_rel.copy() if cache_rel is not None else zeros
        if self._vision_bridge_fallback == "zeros":
            return zeros
        if self._vision_bridge_fallback == "cache_track":
            if cache_rel is not None:
                return cache_rel.copy()
            return track_rel
        raise ValueError(
            "vision_bridge_fallback must be one of 'track', 'cache', 'zeros', 'cache_track'."
        )

    def _fill_obs_buffer_zeros(self) -> None:
        self._obs_buffer.clear()
        for _ in range(self.history_len):
            self._obs_buffer.append(np.zeros(self.single_obs_dim, dtype=np.float32))

    def _reset_camera_cache(self) -> None:
        self._last_camera_frame = None
        self._last_camera_depth = None
        self._last_camera_detections = []
        self._last_detected_rel_gates = [None] * self.lookahead_gates
        self._last_detection_confidences = np.zeros(self.lookahead_gates, dtype=np.float32)

    def _build_camera_params(self, width: int, height: int, fov_deg: float) -> CameraParams:
        fy = 0.5 * float(height) / np.tan(0.5 * np.deg2rad(fov_deg))
        fx = fy
        return CameraParams(
            fx=float(fx),
            fy=float(fy),
            cx=0.5 * float(width),
            cy=0.5 * float(height),
            width=int(width),
            height=int(height),
        )

    def _estimate_gate_physical_width(self) -> float:
        widths = [2.0 * float(gate["radius"]) for track in self._track_options for gate in track]
        return max(0.6, float(np.mean(widths)))

    def _clone_track(self, gates: list[dict]) -> list[dict]:
        return [
            dict(
                center=np.array(g["center"], dtype=np.float64),
                normal=np.array(g["normal"], dtype=np.float64),
                radius=float(g["radius"]),
            )
            for g in gates
        ]

    def _validate_track_options(self) -> None:
        if len(self._track_names) != len(self._track_options):
            raise ValueError("track_names must align with track_options.")
        gate_counts = {len(track) for track in self._track_options}
        if len(gate_counts) != 1:
            raise ValueError("All tracks must currently have the same number of gates.")

    def _select_track_for_reset(self) -> None:
        if self._sample_track_on_reset:
            self._current_track_idx = int(self._track_rng.integers(len(self._track_options)))
        else:
            self._current_track_idx = min(self._current_track_idx, len(self._track_options) - 1)
        self.gates = self._clone_track(self._track_options[self._current_track_idx])
        self.n_gates = len(self.gates)

    def _randomize_start_pose(self) -> None:
        """Spawn behind the first gate with configurable jitter for generalization."""
        gate0 = self.gates[0]
        forward = np.array(gate0["normal"], dtype=np.float64)
        lateral = _horizontal_lateral(forward)

        longitudinal = self._start_distance + self._track_rng.uniform(
            -self._start_longitudinal_jitter, self._start_longitudinal_jitter
        )
        lateral_offset = self._track_rng.uniform(
            -self._start_lateral_jitter, self._start_lateral_jitter
        )
        vertical_offset = self._track_rng.uniform(
            -self._start_vertical_jitter, self._start_vertical_jitter
        )

        start_pos = (
            np.array(gate0["center"], dtype=np.float64)
            - forward * longitudinal
            + lateral * lateral_offset
        )
        start_pos[2] = max(0.4, start_pos[2] + vertical_offset)

        nominal_yaw = float(np.arctan2(forward[1], forward[0]))
        start_yaw = nominal_yaw + self._track_rng.uniform(
            -self._start_yaw_jitter, self._start_yaw_jitter
        )
        quat = p.getQuaternionFromEuler([0.0, 0.0, start_yaw])

        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start_pos,
            quat,
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.CLIENT,
        )

    def _apply_random_disturbance(self) -> None:
        """Apply a lightweight random force/torque impulse for robustness stress tests."""
        if self._disturbance_force_std <= 0.0 and self._disturbance_torque_std <= 0.0:
            return

        if self._disturbance_force_std > 0.0:
            force = self._disturbance_rng.normal(0.0, self._disturbance_force_std, size=3)
            p.applyExternalForce(
                self.DRONE_IDS[0],
                -1,
                forceObj=force.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

        if self._disturbance_torque_std > 0.0:
            torque = self._disturbance_rng.normal(0.0, self._disturbance_torque_std, size=3)
            p.applyExternalTorque(
                self.DRONE_IDS[0],
                -1,
                torqueObj=torque.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

    # ------------------------------------------------------------------
    # Camera + scene helpers
    # ------------------------------------------------------------------

    def _build_default_visual_theme(self) -> dict:
        gate_color = np.array([0.97, 0.62, 0.16, 1.0], dtype=np.float32)
        return {
            "gate_color": gate_color,
            "gate_active_color": np.array([1.0, 0.88, 0.30, 1.0], dtype=np.float32),
            "floor_texture": _FLOOR_TEXTURES[0],
            "inner_floor_texture": _FLOOR_TEXTURES[-1],
            "wall_texture": _WALL_TEXTURES[0],
            "wall_color": np.array([0.72, 0.75, 0.79, 1.0], dtype=np.float32),
            "inner_floor_color": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "clutter_colors": [
                np.array([0.25, 0.31, 0.38, 1.0], dtype=np.float32),
                np.array([0.78, 0.20, 0.18, 1.0], dtype=np.float32),
                np.array([0.17, 0.47, 0.63, 1.0], dtype=np.float32),
            ],
            "light_direction": _normalize(np.array([0.35, -0.45, -1.0], dtype=np.float64)),
            "light_color": np.array([1.0, 0.98, 0.94], dtype=np.float32),
            "ambient_coeff": 0.62,
            "diffuse_coeff": 0.55,
            "specular_coeff": 0.15,
            "exposure": 1.0,
            "color_gain": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "occlusion_count": self._camera_occlusion_count,
        }

    def _sample_visual_theme(self) -> dict:
        if not self._scene_randomization:
            return self._build_default_visual_theme()

        gate_palette = [
            np.array([0.97, 0.62, 0.16, 1.0], dtype=np.float32),
            np.array([0.98, 0.72, 0.18, 1.0], dtype=np.float32),
            np.array([0.91, 0.48, 0.12, 1.0], dtype=np.float32),
        ]
        wall_palette = [
            np.array([0.70, 0.73, 0.77, 1.0], dtype=np.float32),
            np.array([0.64, 0.70, 0.76, 1.0], dtype=np.float32),
            np.array([0.77, 0.78, 0.72, 1.0], dtype=np.float32),
        ]
        gate_color = gate_palette[int(self._visual_rng.integers(len(gate_palette)))]
        light_az = np.deg2rad(self._visual_rng.uniform(-70.0, 70.0))
        light_el = np.deg2rad(self._visual_rng.uniform(-55.0, -20.0))
        light_direction = _normalize(
            np.array(
                [
                    np.cos(light_el) * np.cos(light_az),
                    np.cos(light_el) * np.sin(light_az),
                    np.sin(light_el),
                ],
                dtype=np.float64,
            )
        )
        return {
            "gate_color": gate_color,
            "gate_active_color": np.clip(gate_color + np.array([0.08, 0.08, 0.04, 0.0]), 0.0, 1.0),
            "floor_texture": _FLOOR_TEXTURES[int(self._visual_rng.integers(len(_FLOOR_TEXTURES)))],
            "inner_floor_texture": _FLOOR_TEXTURES[int(self._visual_rng.integers(len(_FLOOR_TEXTURES)))],
            "wall_texture": _WALL_TEXTURES[int(self._visual_rng.integers(len(_WALL_TEXTURES)))],
            "wall_color": wall_palette[int(self._visual_rng.integers(len(wall_palette)))],
            "inner_floor_color": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "clutter_colors": [
                np.array([0.23, 0.30, 0.36, 1.0], dtype=np.float32),
                np.array([0.79, 0.24, 0.20, 1.0], dtype=np.float32),
                np.array([0.16, 0.50, 0.70, 1.0], dtype=np.float32),
            ],
            "light_direction": light_direction,
            "light_color": np.array(
                self._visual_rng.uniform([0.90, 0.92, 0.90], [1.00, 1.00, 0.98]),
                dtype=np.float32,
            ),
            "ambient_coeff": float(self._visual_rng.uniform(0.55, 0.72)),
            "diffuse_coeff": float(self._visual_rng.uniform(0.48, 0.66)),
            "specular_coeff": float(self._visual_rng.uniform(0.10, 0.22)),
            "exposure": float(1.0 + self._visual_rng.uniform(-self._camera_exposure_jitter, self._camera_exposure_jitter)),
            "color_gain": np.array(
                self._visual_rng.uniform(0.94, 1.06, size=3),
                dtype=np.float32,
            ),
            "occlusion_count": max(0, self._camera_occlusion_count + int(self._visual_rng.integers(0, 2))),
        }

    def _clear_visual_handles(self) -> None:
        self._scene_body_ids = []
        self._gate_visual_body_ids = []
        self._texture_ids = {}
        self._debug_text_ids = []

    def _rebuild_scene_visuals(self) -> None:
        self._clear_visual_handles()
        if not self._scene_visuals:
            return
        self._visual_theme = self._sample_visual_theme()
        self._spawn_arena_visuals()
        self._spawn_gate_visuals()

    def _spawn_arena_visuals(self) -> None:
        self._spawn_box(
            half_extents=np.array([13.6, 13.6, 0.04]),
            position=np.array([0.0, 0.0, -0.04]),
            rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            texture_name=self._visual_theme["floor_texture"],
        )
        self._spawn_box(
            half_extents=np.array([8.0, 8.0, 0.015]),
            position=np.array([0.0, 0.0, -0.005]),
            rgba=self._visual_theme["inner_floor_color"],
            texture_name=self._visual_theme["inner_floor_texture"],
        )

        wall_color = self._visual_theme["wall_color"]
        wall_texture = self._visual_theme["wall_texture"]
        wall_specs = [
            (np.array([13.6, 0.12, 1.8]), np.array([0.0, 13.55, 1.8])),
            (np.array([13.6, 0.12, 1.8]), np.array([0.0, -13.55, 1.8])),
            (np.array([0.12, 13.6, 1.8]), np.array([13.55, 0.0, 1.8])),
            (np.array([0.12, 13.6, 1.8]), np.array([-13.55, 0.0, 1.8])),
        ]
        for half_extents, position in wall_specs:
            self._spawn_box(
                half_extents=half_extents,
                position=position,
                rgba=wall_color,
                texture_name=wall_texture,
            )

        for _ in range(self._scene_clutter_count):
            side = int(self._visual_rng.integers(4))
            if side == 0:
                x = self._visual_rng.uniform(-10.5, 10.5)
                y = self._visual_rng.uniform(8.5, 12.2)
            elif side == 1:
                x = self._visual_rng.uniform(-10.5, 10.5)
                y = self._visual_rng.uniform(-12.2, -8.5)
            elif side == 2:
                x = self._visual_rng.uniform(8.5, 12.2)
                y = self._visual_rng.uniform(-10.5, 10.5)
            else:
                x = self._visual_rng.uniform(-12.2, -8.5)
                y = self._visual_rng.uniform(-10.5, 10.5)

            width = float(self._visual_rng.uniform(0.18, 0.55))
            depth = float(self._visual_rng.uniform(0.18, 0.55))
            height = float(self._visual_rng.uniform(0.4, 1.8))
            color = self._visual_theme["clutter_colors"][
                int(self._visual_rng.integers(len(self._visual_theme["clutter_colors"])))
            ]
            self._spawn_box(
                half_extents=np.array([width, depth, height]),
                position=np.array([x, y, height]),
                rgba=color,
            )

    def _spawn_gate_visuals(self) -> None:
        gate_color = self._visual_theme["gate_color"]
        for gate in self.gates:
            gate_ids: list[int] = []
            normal = _normalize(gate["normal"])
            lateral = _horizontal_lateral(normal)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            radius = float(gate["radius"])
            frame = max(0.06, 0.16 * radius)
            depth = max(0.04, 0.5 * frame)

            vertical_half_extents = np.array([radius + frame, depth, 0.5 * frame], dtype=np.float64)
            vertical_quat = _quat_from_axes(up, normal, lateral)
            for side in (-1.0, 1.0):
                pos = gate["center"] + side * lateral * (radius + 0.5 * frame)
                gate_ids.append(
                    self._spawn_box(
                        half_extents=vertical_half_extents,
                        position=pos,
                        rgba=gate_color,
                        orientation=vertical_quat,
                    )
                )

            # Rotate the top/bottom rails 90 degrees about their long axis so the
            # broad face sits in the gate plane instead of facing upward.
            horizontal_half_extents = np.array([radius + frame, 0.5 * frame, depth], dtype=np.float64)
            horizontal_quat = _quat_from_axes(lateral, up, normal)
            for side in (-1.0, 1.0):
                pos = gate["center"] + side * up * (radius + 0.5 * frame)
                gate_ids.append(
                    self._spawn_box(
                        half_extents=horizontal_half_extents,
                        position=pos,
                        rgba=gate_color,
                        orientation=horizontal_quat,
                    )
                )

            self._gate_visual_body_ids.append(gate_ids)

        self._apply_gate_highlights()

    def _spawn_box(
        self,
        half_extents: np.ndarray,
        position: np.ndarray,
        rgba: np.ndarray,
        orientation: tuple[float, float, float, float] | None = None,
        texture_name: str | None = None,
    ) -> int:
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=np.asarray(half_extents, dtype=np.float64).tolist(),
            rgbaColor=np.asarray(rgba, dtype=np.float64).tolist(),
            physicsClientId=self.CLIENT,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=np.asarray(position, dtype=np.float64).tolist(),
            baseOrientation=orientation if orientation is not None else [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.CLIENT,
        )
        if texture_name is not None:
            tex_id = self._load_texture(texture_name)
            if tex_id >= 0:
                p.changeVisualShape(body_id, -1, textureUniqueId=tex_id, physicsClientId=self.CLIENT)
        self._scene_body_ids.append(body_id)
        return body_id

    def _load_texture(self, texture_name: str) -> int:
        if texture_name in self._texture_ids:
            return self._texture_ids[texture_name]
        texture_path = _TEXTURE_DIR / texture_name
        if not texture_path.exists():
            self._texture_ids[texture_name] = -1
            return -1
        tex_id = p.loadTexture(str(texture_path), physicsClientId=self.CLIENT)
        self._texture_ids[texture_name] = tex_id
        return tex_id

    def _apply_gate_highlights(self) -> None:
        if not self._gate_visual_body_ids:
            return
        passive = self._visual_theme["gate_color"]
        active = self._visual_theme["gate_active_color"]
        for idx, gate_ids in enumerate(self._gate_visual_body_ids):
            color = active if idx == self._next_gate else passive
            for body_id in gate_ids:
                p.changeVisualShape(
                    body_id,
                    -1,
                    rgbaColor=np.asarray(color, dtype=np.float64).tolist(),
                    physicsClientId=self.CLIENT,
                )

    def _render_onboard_camera(self) -> Optional[np.ndarray]:
        if not self._camera_active:
            return None

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        forward = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tilt = np.deg2rad(self._camera_tilt_deg)
        cam_forward = _normalize(np.cos(tilt) * forward - np.sin(tilt) * up)
        cam_up = _normalize(np.sin(tilt) * forward + np.cos(tilt) * up)
        eye = pos + forward * self._camera_offset_forward + up * self._camera_offset_up
        target = eye + cam_forward * self._camera_far

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=cam_up.tolist(),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self._camera_fov,
            aspect=float(self._camera_width) / float(self._camera_height),
            nearVal=self._camera_near,
            farVal=self._camera_far,
        )

        renderer = p.ER_BULLET_HARDWARE_OPENGL if self._gui_enabled else p.ER_TINY_RENDERER
        render_kwargs = dict(
            width=self._camera_width,
            height=self._camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            physicsClientId=self.CLIENT,
        )
        light_kwargs = dict(
            shadow=1 if self._camera_shadow else 0,
            lightDirection=np.asarray(self._visual_theme["light_direction"], dtype=np.float64).tolist(),
            lightColor=np.asarray(self._visual_theme["light_color"], dtype=np.float64).tolist(),
            lightAmbientCoeff=float(self._visual_theme["ambient_coeff"]),
            lightDiffuseCoeff=float(self._visual_theme["diffuse_coeff"]),
            lightSpecularCoeff=float(self._visual_theme["specular_coeff"]),
        )

        try:
            _, _, rgba, depth, _ = p.getCameraImage(**render_kwargs, **light_kwargs)
        except TypeError:
            _, _, rgba, depth, _ = p.getCameraImage(**render_kwargs)

        rgba_np = np.asarray(rgba, dtype=np.uint8).reshape(self._camera_height, self._camera_width, 4)
        rgb = rgba_np[:, :, :3]
        rgb = self._postprocess_camera_frame(rgb)

        self._last_camera_frame = rgb
        self._last_camera_depth = np.asarray(depth).reshape(self._camera_height, self._camera_width)
        return rgb

    def _postprocess_camera_frame(self, rgb: np.ndarray) -> np.ndarray:
        image = rgb.astype(np.float32)
        image *= float(self._visual_theme["exposure"])
        image *= np.asarray(self._visual_theme["color_gain"], dtype=np.float32)

        if self._camera_noise_std > 0.0:
            image += self._camera_rng.normal(0.0, self._camera_noise_std, size=image.shape)

        if self._camera_occlusion_prob > 0.0 and self._camera_rng.random() < self._camera_occlusion_prob:
            occ_count = max(1, int(self._visual_theme.get("occlusion_count", self._camera_occlusion_count)))
            for _ in range(occ_count):
                occ_w = int(self._camera_rng.integers(max(8, self._camera_width // 12), max(12, self._camera_width // 4)))
                occ_h = int(self._camera_rng.integers(max(8, self._camera_height // 12), max(12, self._camera_height // 3)))
                x0 = int(self._camera_rng.integers(0, max(1, self._camera_width - occ_w)))
                y0 = int(self._camera_rng.integers(0, max(1, self._camera_height - occ_h)))
                shade = float(self._camera_rng.uniform(18.0, 60.0))
                image[y0 : y0 + occ_h, x0 : x0 + occ_w] = shade

        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------

    def _setup_debug_visuals(self) -> None:
        """Draw lightweight gate labels the first time the GUI is used."""
        if self._debug_text_ids:
            return
        for idx, gate in enumerate(self.gates):
            color = [0.1, 0.8, 0.1] if idx == 0 else [0.9, 0.8, 0.2]
            text_pos = gate["center"] + np.array([0.0, 0.0, 0.45])
            text_id = p.addUserDebugText(
                text=f"G{idx}",
                textPosition=text_pos,
                textColorRGB=color,
                textSize=1.3,
                lifeTime=0,
                physicsClientId=self.CLIENT,
            )
            self._debug_text_ids.append(text_id)

    def _update_debug_visuals(self) -> None:
        """Keep the chase camera and gate labels aligned with episode state."""
        if self._camera_follow:
            drone_pos = self.pos[0]
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
                cameraTargetPosition=drone_pos,
                physicsClientId=self.CLIENT,
            )

        self._apply_gate_highlights()
        if self._debug_text_ids and len(self._debug_text_ids) == self.n_gates:
            for idx, gate in enumerate(self.gates):
                color = [0.1, 0.95, 0.1] if idx == self._next_gate else [0.9, 0.8, 0.2]
                text_pos = gate["center"] + np.array([0.0, 0.0, 0.45])
                p.addUserDebugText(
                    text=f"G{idx}",
                    textPosition=text_pos,
                    textColorRGB=color,
                    textSize=1.3,
                    lifeTime=0,
                    replaceItemUniqueId=self._debug_text_ids[idx],
                    physicsClientId=self.CLIENT,
                )

    # ------------------------------------------------------------------
    # Gate crossing detection
    # ------------------------------------------------------------------

    def _get_gate_sides(self) -> np.ndarray:
        pos = self.pos[0]
        sides = np.empty(self.n_gates)
        for i, g in enumerate(self.gates):
            d = float(np.dot(pos - g["center"], g["normal"]))
            sides[i] = 1.0 if d >= 0.0 else -1.0
        return sides

    def _detect_gate_crossing(self, prev_sides: np.ndarray, curr_sides: np.ndarray) -> dict:
        events = {"gate_passed": False, "gate_id": -1, "wrong_way": False, "collision": False}

        i = self._next_gate
        gate = self.gates[i]

        if prev_sides[i] != curr_sides[i]:
            d1 = np.dot(self._prev_pos - gate["center"], gate["normal"])
            d2 = np.dot(self.pos[0] - gate["center"], gate["normal"])
            denom = abs(d1) + abs(d2)
            t = abs(d1) / denom if denom > 1e-9 else 0.5
            crossing = self._prev_pos + t * (self.pos[0] - self._prev_pos)

            diff = crossing - gate["center"]
            in_plane = diff - np.dot(diff, gate["normal"]) * gate["normal"]
            if np.linalg.norm(in_plane) < gate["radius"]:
                events["gate_id"] = i
                if prev_sides[i] < 0.0 and curr_sides[i] > 0.0:
                    events["gate_passed"] = True
                else:
                    events["wrong_way"] = True

        # Crash proxy: tilt > 60 deg
        rpy = self.rpy[0]
        if abs(rpy[0]) > np.pi / 3 or abs(rpy[1]) > np.pi / 3:
            events["collision"] = True

        return events


def make_env(cfg: dict, gui: bool = False, camera_follow: bool = False) -> GateRaceAviary:
    """Build a GateRaceAviary from a config dict (env sub-dict)."""
    track_names = list(cfg.get("track_names", []))
    if "track_name" in cfg:
        track_names = [cfg["track_name"]]

    if track_names:
        track_options = get_tracks(track_names)
        gates = track_options[0]
    else:
        if cfg.get("gates") is None:
            gates = get_track("rect_default")
        else:
            gates = [
                dict(
                    center=np.array(g["center"], dtype=np.float64),
                    normal=np.array(g["normal"], dtype=np.float64),
                    radius=float(g["radius"]),
                )
                for g in cfg.get("gates", DEFAULT_GATES)
            ]
        track_options = [gates]

    camera_cfg = dict(cfg.get("camera", {}))
    visual_cfg = dict(cfg.get("visual", {}))

    return GateRaceAviary(
        gates=gates,
        track_options=track_options,
        track_names=track_names if track_names else ["custom"],
        sample_track_on_reset=bool(cfg.get("sample_track_on_reset", False)),
        ctrl_freq=int(cfg.get("ctrl_freq", 20)),
        pyb_freq=int(cfg.get("pyb_freq", 240)),
        clip_radius=float(cfg.get("clip_radius_start", 0.25)),
        max_dyaw=float(cfg.get("max_dyaw", 0.3)),
        episode_len_sec=float(cfg.get("episode_len_sec", 12.0)),
        history_len=int(cfg.get("history_len", 3)),
        lookahead_gates=int(cfg.get("lookahead_gates", 2)),
        include_gate_normals=bool(cfg.get("include_gate_normals", False)),
        include_relative_heading=bool(cfg.get("include_relative_heading", False)),
        smooth_coef=float(cfg.get("smooth_coef", 0.01)),
        progress_coef=float(cfg.get("progress_coef", 0.1)),
        gui=gui,
        camera_follow=camera_follow,
        camera_distance=float(cfg.get("camera_distance", 3.2)),
        camera_yaw=float(cfg.get("camera_yaw", -35.0)),
        camera_pitch=float(cfg.get("camera_pitch", -28.0)),
        start_distance=float(cfg.get("start_distance", 3.0)),
        start_longitudinal_jitter=float(cfg.get("start_longitudinal_jitter", 0.0)),
        start_lateral_jitter=float(cfg.get("start_lateral_jitter", 0.0)),
        start_vertical_jitter=float(cfg.get("start_vertical_jitter", 0.0)),
        start_yaw_jitter=float(cfg.get("start_yaw_jitter", 0.0)),
        disturbance_force_std=float(cfg.get("disturbance_force_std", 0.0)),
        disturbance_torque_std=float(cfg.get("disturbance_torque_std", 0.0)),
        observation_source=str(cfg.get("observation_source", "state")),
        vision_bridge_fallback=str(cfg.get("vision_bridge_fallback", "cache_track")),
        vision_bridge_normal_source=str(cfg.get("vision_bridge_normal_source", "track")),
        vision_bridge_min_confidence=float(cfg.get("vision_bridge_min_confidence", 0.1)),
        enable_onboard_camera=bool(camera_cfg.get("enabled", cfg.get("enable_onboard_camera", False))),
        camera_width=int(camera_cfg.get("width", cfg.get("camera_width", 128))),
        camera_height=int(camera_cfg.get("height", cfg.get("camera_height", 96))),
        camera_fov=float(camera_cfg.get("fov", cfg.get("camera_fov", 85.0))),
        camera_tilt_deg=float(camera_cfg.get("tilt_deg", cfg.get("camera_tilt_deg", 8.0))),
        camera_offset_forward=float(camera_cfg.get("offset_forward", cfg.get("camera_offset_forward", 0.08))),
        camera_offset_up=float(camera_cfg.get("offset_up", cfg.get("camera_offset_up", 0.03))),
        camera_near=float(camera_cfg.get("near", cfg.get("camera_near", 0.03))),
        camera_far=float(camera_cfg.get("far", cfg.get("camera_far", 25.0))),
        camera_noise_std=float(camera_cfg.get("noise_std", cfg.get("camera_noise_std", 0.0))),
        camera_exposure_jitter=float(camera_cfg.get("exposure_jitter", cfg.get("camera_exposure_jitter", 0.0))),
        camera_occlusion_prob=float(camera_cfg.get("occlusion_prob", cfg.get("camera_occlusion_prob", 0.0))),
        camera_occlusion_count=int(camera_cfg.get("occlusion_count", cfg.get("camera_occlusion_count", 1))),
        camera_include_in_info=bool(camera_cfg.get("include_in_info", cfg.get("camera_include_in_info", False))),
        camera_shadow=bool(camera_cfg.get("shadow", cfg.get("camera_shadow", True))),
        scene_visuals=bool(visual_cfg.get("enabled", cfg.get("scene_visuals", False))),
        scene_randomization=bool(visual_cfg.get("randomize", cfg.get("scene_randomization", False))),
        scene_clutter_count=int(visual_cfg.get("clutter_count", cfg.get("scene_clutter_count", 8))),
    )

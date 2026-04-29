from collections import deque

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

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

_BASE_OBS_DIM = 12  # per timestep: vel_body(3) + ang_body(3) + rel_g0(3) + rel_g1(3)


class GateRaceAviary(BaseAviary):
    """Single-drone gate racing environment.

    Observation (history_len × 12 dims, default 36):
        Per frame: body-frame velocity (3), body-frame angular rate (3),
        relative position of next 2 gates in body frame (3+3).
        Last `history_len` frames are concatenated.

    Action (4-dim, normalised [-1, 1]):
        [dx_b, dy_b, dz_b, d_yaw] — body-frame position delta + yaw delta,
        scaled by clip_radius and max_dyaw.  Converted to a world-frame
        waypoint target tracked by DSLPIDControl.

    Reward shaping:
        Dense: velocity alignment toward next gate (dot-product projection).
        Smooth: penalty on action jerk ||a_t - a_{t-1}||.
        Sparse: +10 per correct gate pass, −50 on crash.
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
    ):
        base_track = gates if gates is not None else DEFAULT_GATES
        self._track_options = track_options or [base_track]
        self._track_names = track_names or [f"track_{i}" for i in range(len(self._track_options))]
        self._sample_track_on_reset = bool(sample_track_on_reset and len(self._track_options) > 1)
        self._track_rng = np.random.default_rng()
        self._current_track_idx = 0
        self.gates = self._clone_track(self._track_options[self._current_track_idx])
        self._validate_track_options()
        self.n_gates = len(self.gates)
        self.clip_radius = clip_radius
        self.max_dyaw = max_dyaw
        self._episode_len_sec = episode_len_sec
        self.history_len = history_len
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
        self._debug_text_ids: list[int] = []

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

    def set_clip_radius(self, r: float) -> None:
        """Update clip_radius mid-training (curriculum)."""
        self.clip_radius = float(r)

    def get_full_state(self) -> dict:
        """Full world-frame state dict used by the expert and DCL adapter."""
        state = self._getDroneStateVector(0)
        return {
            "pos":        state[0:3].copy(),
            "quat":       state[3:7].copy(),
            "rpy":        state[7:10].copy(),
            "vel":        state[10:13].copy(),
            "ang_vel":    state[13:16].copy(),
            "next_gate":  self._next_gate,
            "clip_radius": self.clip_radius,
            "max_dyaw":   self.max_dyaw,
            "track_name": self.track_name,
        }

    @property
    def track_name(self) -> str:
        return self._track_names[self._current_track_idx]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._track_rng = np.random.default_rng(seed)
        self._select_track_for_reset()
        self.pid.reset()
        self._next_gate = 0
        self._gates_passed = 0
        self._last_gate_events = {}
        self._prev_pos = _INIT_XYZ[0].copy()
        self._last_action = None
        self._current_action = np.zeros(4)
        # Pre-fill history with zeros so the buffer is ready when
        # super().reset() calls _computeObs().
        self._fill_obs_buffer_zeros()
        _, _ = super().reset(seed=seed, options=options)
        self._randomize_start_pose()
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
        dim = _BASE_OBS_DIM * self.history_len
        return spaces.Box(
            low=-np.inf * np.ones(dim, dtype=np.float32),
            high=np.inf * np.ones(dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()[:4]
        self._current_action = action.astype(np.float32)  # store for smoothness

        state = self._getDroneStateVector(0)
        pos     = state[0:3]
        quat    = state[3:7]
        vel     = state[10:13]
        ang_vel = state[13:16]
        yaw     = float(state[9])

        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)  # body→world

        delta_pos_body = action[:3] * self.clip_radius
        delta_yaw      = float(action[3]) * self.max_dyaw

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
        pos  = self.pos[0]
        vel  = self.vel[0]   # world frame
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
        info["next_gate"]    = self._next_gate
        info["clip_radius"]  = self.clip_radius
        info["track_name"]   = self.track_name
        return info

    # ------------------------------------------------------------------
    # Internal observation helpers
    # ------------------------------------------------------------------

    def _compute_single_obs(self) -> np.ndarray:
        """12-dim single-timestep observation."""
        state       = self._getDroneStateVector(0)
        pos         = state[0:3]
        quat        = state[3:7]
        vel_world   = state[10:13]
        ang_world   = state[13:16]

        rot_T = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3).T  # world→body

        vel_body = rot_T @ vel_world
        ang_body = rot_T @ ang_world

        g0 = self.gates[self._next_gate % self.n_gates]["center"]
        g1 = self.gates[(self._next_gate + 1) % self.n_gates]["center"]
        rel_g0 = rot_T @ (g0 - pos)
        rel_g1 = rot_T @ (g1 - pos)

        return np.concatenate([vel_body, ang_body, rel_g0, rel_g1]).astype(np.float32)

    def _fill_obs_buffer_zeros(self) -> None:
        self._obs_buffer.clear()
        for _ in range(self.history_len):
            self._obs_buffer.append(np.zeros(_BASE_OBS_DIM, dtype=np.float32))

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
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
        lateral_norm = np.linalg.norm(lateral[:2])
        if lateral_norm < 1e-9:
            lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            lateral /= lateral_norm

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
        pos   = self.pos[0]
        sides = np.empty(self.n_gates)
        for i, g in enumerate(self.gates):
            d = float(np.dot(pos - g["center"], g["normal"]))
            sides[i] = 1.0 if d >= 0.0 else -1.0
        return sides

    def _detect_gate_crossing(self, prev_sides: np.ndarray, curr_sides: np.ndarray) -> dict:
        events = {"gate_passed": False, "gate_id": -1, "wrong_way": False, "collision": False}

        i    = self._next_gate
        gate = self.gates[i]

        if prev_sides[i] != curr_sides[i]:
            d1    = np.dot(self._prev_pos - gate["center"], gate["normal"])
            d2    = np.dot(self.pos[0]    - gate["center"], gate["normal"])
            denom = abs(d1) + abs(d2)
            t     = abs(d1) / denom if denom > 1e-9 else 0.5
            crossing = self._prev_pos + t * (self.pos[0] - self._prev_pos)

            diff     = crossing - gate["center"]
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
    )

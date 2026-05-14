"""Lookahead expert policy for gate racing.

Targets a point slightly beyond the next gate while blending future gate
directions and normals, so the expert begins setting up the turn earlier
than a pure gate-center chaser and handles longer courses better.
"""

import numpy as np
import pybullet as p


class ExpertPolicy:
    """Lookahead racing expert using gate normals and future-gate geometry.

    At each step the expert reads the current environment state, computes
    a target point just beyond the next gate plane while anticipating the
    next several turns, and converts that world-frame target into the same
    normalised body-frame waypoint-delta action used by the learner.
    """

    def act(self, env) -> np.ndarray:
        """Return a normalized action for the current env state.

        Parameters
        ----------
        env : GateRaceAviary
            Live environment; state is read via env.get_full_state().

        Returns
        -------
        action : ndarray, shape (4,), values in [-1, 1]
        """
        st = env.get_full_state()
        pos = st["pos"]
        quat = st["quat"]
        yaw = float(st["rpy"][2])
        clip_radius = st["clip_radius"]
        max_dyaw = st["max_dyaw"]

        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)  # body→world

        next_gate = int(st["next_gate"])
        gates = env.gates
        target_world, desired_dir = _lookahead_target(gates, next_gate)

        delta_world = target_world - pos
        delta_body = rot.T @ delta_world

        # Clip to clip_radius and normalise to [-1, 1]
        norm = float(np.linalg.norm(delta_body))
        if norm > clip_radius:
            delta_body = delta_body * (clip_radius / norm)
        action_xyz = delta_body / clip_radius

        # Desired yaw: align with the blended pass-through / exit direction.
        desired_yaw = float(np.arctan2(desired_dir[1], desired_dir[0]))
        delta_yaw = _wrap_angle(desired_yaw - yaw)
        action_yaw = float(np.clip(delta_yaw / max_dyaw, -1.0, 1.0))

        return np.array([*action_xyz, action_yaw], dtype=np.float32)


def _lookahead_target(gates: list[dict], next_gate_idx: int, horizon: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return a target point and desired heading direction for the expert.

    The target is built from the next gate's pass-through normal plus a
    decaying blend of upcoming segment directions and gate normals.
    That makes the expert less tied to simple 4-gate loops and gives it a
    longer planning horizon on 5- to 6-gate courses.
    """
    horizon = max(3, min(int(horizon), len(gates)))
    future = [gates[(next_gate_idx + i) % len(gates)] for i in range(horizon)]
    g0 = future[0]

    seg_dirs = [
        _safe_unit(future[i + 1]["center"] - future[i]["center"], future[i]["normal"])
        for i in range(len(future) - 1)
    ]

    blended = 1.7 * g0["normal"]
    for idx, seg in enumerate(seg_dirs):
        blended += (0.95 * (0.62 ** idx)) * seg
    for idx, gate in enumerate(future[1:]):
        blended += (0.45 * (0.55 ** idx)) * gate["normal"]
    exit_dir = _safe_unit(blended, g0["normal"])

    path_dir = _safe_unit(sum((0.85 * (0.65 ** idx)) * seg for idx, seg in enumerate(seg_dirs)), g0["normal"])
    lateral_hint = path_dir - np.dot(path_dir, g0["normal"]) * g0["normal"]
    lateral_hint = _safe_unit(lateral_hint, np.zeros(3))

    straightness = 0.0
    if len(seg_dirs) >= 2:
        straightness = float(np.clip(np.dot(seg_dirs[0], seg_dirs[1]), -1.0, 1.0))
    gate_scale = float(g0.get("radius", 0.75)) / 0.75
    pass_offset = gate_scale * (0.28 + 0.08 * max(0.0, straightness))
    lateral_offset = gate_scale * (0.10 + 0.10 * (1.0 - max(0.0, straightness)))
    cruise_bonus = 0.10 * gate_scale * max(0.0, straightness)

    target_world = (
        g0["center"].copy()
        + exit_dir * pass_offset
        + lateral_hint * lateral_offset
        + path_dir * cruise_bonus
    )
    return target_world, exit_dir


def _safe_unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-9:
            return np.zeros_like(vec, dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64) / fallback_norm
    return np.asarray(vec, dtype=np.float64) / norm


def _wrap_angle(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a

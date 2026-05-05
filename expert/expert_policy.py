"""Lookahead expert policy for gate racing.

Targets a point slightly beyond the next gate while blending in the next
two gate directions and normals, so the expert begins setting up the
turn earlier than a pure gate-center chaser.
"""

import numpy as np
import pybullet as p


class ExpertPolicy:
    """Lookahead racing expert using gate normals and future-gate geometry.

    At each step the expert reads the current environment state, computes
    a target point just beyond the next gate plane while anticipating the
    next two turns, and converts that world-frame target into the same
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


def _lookahead_target(gates: list[dict], next_gate_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a target point and desired heading direction for the expert."""
    g0 = gates[next_gate_idx % len(gates)]
    g1 = gates[(next_gate_idx + 1) % len(gates)]
    g2 = gates[(next_gate_idx + 2) % len(gates)]

    seg01 = _safe_unit(g1["center"] - g0["center"], g0["normal"])
    seg12 = _safe_unit(g2["center"] - g1["center"], g1["normal"])

    exit_dir = _safe_unit(
        1.8 * g0["normal"] + 0.9 * seg01 + 0.45 * seg12 + 0.35 * g1["normal"],
        g0["normal"],
    )
    lateral_hint = seg01 - np.dot(seg01, g0["normal"]) * g0["normal"]
    lateral_hint = _safe_unit(lateral_hint, np.zeros(3))

    target_world = (
        g0["center"].copy()
        + exit_dir * 0.32
        + lateral_hint * 0.12
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

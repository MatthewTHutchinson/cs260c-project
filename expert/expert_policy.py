"""Gate-to-gate expert policy for gate racing.

Directly targets the next gate center with a small lookahead offset,
computing a body-frame waypoint-delta action identical to the learner's
action space.
"""

import numpy as np
import pybullet as p


class ExpertPolicy:
    """Gate-chasing expert: fly straight toward each gate center in sequence.

    At each step the expert reads the current environment state, aims at the
    next gate center, and converts the world-frame delta to a normalised
    body-frame waypoint-delta action.
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

        # Aim at the next gate center
        next_gate = int(st["next_gate"])
        gates = env.gates
        target_world = gates[next_gate]["center"].copy()
        gate_normal = gates[next_gate]["normal"]

        # Push target slightly past the gate plane so the drone passes through
        target_world = target_world + gate_normal * 0.3

        delta_world = target_world - pos
        delta_body = rot.T @ delta_world

        # Clip to clip_radius and normalise to [-1, 1]
        norm = float(np.linalg.norm(delta_body))
        if norm > clip_radius:
            delta_body = delta_body * (clip_radius / norm)
        action_xyz = delta_body / clip_radius

        # Desired yaw: face toward the gate
        desired_yaw = float(np.arctan2(delta_world[1], delta_world[0]))
        delta_yaw = _wrap_angle(desired_yaw - yaw)
        action_yaw = float(np.clip(delta_yaw / max_dyaw, -1.0, 1.0))

        return np.array([*action_xyz, action_yaw], dtype=np.float32)


def _wrap_angle(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a

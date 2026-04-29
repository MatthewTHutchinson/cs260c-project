"""Abstract drone racing environment interface.

Every backend — gym-pybullet-drones today, DCL platform tomorrow — must
implement this interface.  The policy, expert, and training code only
reference AbstractDroneRacingEnv, so swapping backends requires editing
exactly one file (the adapter).
"""

from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class AbstractDroneRacingEnv(ABC, gym.Env):
    """Minimal contract for a drone racing environment.

    Observation layout (history_len × 12 dims):
        Per frame: vel_body(3), ang_body(3), rel_gate0_body(3), rel_gate1_body(3).
        Frames are concatenated oldest→newest.

    Action layout (4 dims, normalised [-1, 1]):
        [dx_b, dy_b, dz_b, d_yaw] — body-frame waypoint delta + yaw rate.
    """

    # --- Gym methods (must be implemented) ---

    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]: ...

    @abstractmethod
    def close(self) -> None: ...

    # --- Domain-specific interface ---

    @abstractmethod
    def get_full_state(self) -> dict:
        """Return a dict with at least: pos, quat, rpy, vel, ang_vel, next_gate,
        clip_radius, max_dyaw.  Used by the expert and the DCL adapter."""
        ...

    @property
    @abstractmethod
    def n_gates_total(self) -> int:
        """Total number of gates in the current track layout."""
        ...

    @abstractmethod
    def set_clip_radius(self, r: float) -> None:
        """Curriculum hook: update the action clip radius mid-training."""
        ...

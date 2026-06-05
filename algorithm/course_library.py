"""Debug course geometry used for privileged teacher data generation.

These course definitions mirror the local practice harness patch. They are
privileged debug geometry and must not be used as runtime inputs to
`AutonomousRacingPilot`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CourseGate:
    index: int
    center_enu_m: tuple[float, float, float]
    yaw_deg: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.center_enu_m, dtype=np.float64)

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def normal_enu(self) -> np.ndarray:
        """Gate plane normal in ENU; yaw zero points along +X."""
        return np.asarray(
            [math.cos(self.yaw_rad), math.sin(self.yaw_rad), 0.0],
            dtype=np.float64,
        )


EASY_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8)),
    CourseGate(1, (20.0, 0.0, 1.8)),
    CourseGate(2, (30.0, 0.0, 1.8)),
)

LATERAL_SOFT_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8)),
    CourseGate(1, (20.0, 0.35, 1.8)),
    CourseGate(2, (30.0, -0.15, 1.8)),
)

LOW_HIGH_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8)),
    CourseGate(1, (20.0, 0.0, 1.6)),
    CourseGate(2, (30.0, 0.0, 1.9)),
)

FOUR_GATE_STRAIGHT_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8)),
    CourseGate(1, (20.0, 0.0, 1.8)),
    CourseGate(2, (30.0, 0.0, 1.8)),
    CourseGate(3, (40.0, 0.0, 1.8)),
)

CIRCULAR_ARC_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8), yaw_deg=0.0),
    CourseGate(1, (18.0, 1.2, 1.8), yaw_deg=8.0),
    CourseGate(2, (26.0, 3.6, 1.8), yaw_deg=16.0),
    CourseGate(3, (33.5, 7.3, 1.8), yaw_deg=25.0),
)

S_CURVE_COURSE = (
    CourseGate(0, (10.0, 0.0, 1.8), yaw_deg=0.0),
    CourseGate(1, (18.0, 1.4, 1.8), yaw_deg=10.0),
    CourseGate(2, (26.0, -1.4, 1.8), yaw_deg=-10.0),
    CourseGate(3, (34.0, 1.5, 1.8), yaw_deg=11.0),
    CourseGate(4, (42.0, -1.2, 1.8), yaw_deg=-9.0),
)

COURSES: dict[str, tuple[CourseGate, ...]] = {
    "easy": EASY_COURSE,
    "lateral_soft": LATERAL_SOFT_COURSE,
    "low_high": LOW_HIGH_COURSE,
    "four_gate_straight": FOUR_GATE_STRAIGHT_COURSE,
    "circular": CIRCULAR_ARC_COURSE,
    "circular_arc": CIRCULAR_ARC_COURSE,
    "s_curve": S_CURVE_COURSE,
}


def course_by_name(name: str) -> tuple[CourseGate, ...]:
    key = name.strip().lower()
    if key not in COURSES:
        available = ", ".join(sorted(COURSES))
        raise ValueError(f"unknown course '{name}', available: {available}")
    return COURSES[key]


def course_names() -> tuple[str, ...]:
    return tuple(sorted(COURSES))


def centers_for_course(gates: Iterable[CourseGate]) -> np.ndarray:
    return np.asarray([gate.center for gate in gates], dtype=np.float64)


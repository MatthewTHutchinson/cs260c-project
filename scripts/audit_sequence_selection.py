#!/usr/bin/env python3
"""Audit distance-aware gate sequencing without simulator gate truth."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_PYTHON = Path(
    os.environ.get(
        "PROJECT_PYTHON",
        "/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python",
    )
)

if (
    os.environ.get("CS260C_NO_PYTHON_REEXEC") != "1"
    and os.environ.get("CS260C_PYTHON_REEXECED") != "1"
    and PROJECT_PYTHON.exists()
    and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve()
):
    os.environ["CS260C_PYTHON_REEXECED"] = "1"
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from algorithm.gate_detector import CameraParams, GateObservation
from algorithm.gate_tracker import GateTracker
from algorithm.types import TrackMode


CAM = CameraParams()
FRAME = np.zeros((CAM.height, CAM.width, 3), dtype=np.uint8)


class ScriptedDetector:
    """Detector test double that returns preplanned FPV-derived candidates."""

    gate_w = 2.7

    def __init__(self, batches: list[list[GateObservation]]) -> None:
        self.batches = batches
        self.calls = 0

    def detect(
        self,
        bgr_frame: np.ndarray,
        cam: CameraParams,
        max_gates: int = 2,
    ) -> list[GateObservation]:
        del bgr_frame, cam
        index = min(self.calls, len(self.batches) - 1)
        self.calls += 1
        return self.batches[index][:max_gates]


def obs(
    distance_m: float,
    *,
    x: float = 320.0,
    y: float = 180.0,
    confidence: float = 0.90,
) -> GateObservation:
    return GateObservation(
        bearing_h=float(np.arctan2(x - CAM.cx, CAM.fx)),
        bearing_v=float(np.arctan2(CAM.cy - y, CAM.fy)),
        distance_est=distance_m,
        confidence=confidence,
        pixel_centre=(x, y),
        source="scripted",
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_multi_candidate_switch() -> list[str]:
    tracker = GateTracker(
        detector=ScriptedDetector(
            [
                [obs(4.0, x=330.0)],
                [obs(1.35, x=320.0)],
                [
                    obs(1.05, x=628.0, y=338.0, confidence=0.80),
                    obs(4.20, x=316.0, y=178.0, confidence=0.92),
                ],
                [obs(1.10, x=630.0, y=340.0, confidence=0.95)],
                [obs(4.05, x=305.0, y=181.0, confidence=0.90)],
            ]
        ),
        camera_params=CAM,
    )

    estimates = [
        tracker.update(FRAME, timestamp_s=0.00),
        tracker.update(FRAME, timestamp_s=0.04),
        tracker.update(FRAME, timestamp_s=0.08),
        tracker.update(FRAME, timestamp_s=0.12),
        tracker.update(FRAME, timestamp_s=0.90),
    ]

    assert_true(estimates[0].sequence_index == 0, "first target should be sequence 0")
    assert_true(estimates[1].mode == TrackMode.COMMIT, "near gate should enter commit")
    assert_true(estimates[2].sequence_index == 1, "far candidate should advance sequence")
    assert_true(estimates[2].distance_m is not None and estimates[2].distance_m > 4.0,
                "far next gate should beat stale close edge gate")
    assert_true(estimates[3].mode == TrackMode.TRACKED, "post-pass close-only detection should be ignored")
    assert_true(estimates[3].sequence_index == 1, "tracked fallback should preserve sequence")
    assert_true(estimates[4].mode == TrackMode.DETECTED, "far gate should reacquire after cooldown")
    assert_true(estimates[4].sequence_index == 1, "reacquired next gate should remain sequence 1")

    return [
        "scenario=multi_candidate_after_commit status=PASS",
        (
            "selected_after_pass="
            f"seq{estimates[2].sequence_index} "
            f"mode={estimates[2].mode.value} "
            f"distance={estimates[2].distance_m:.2f}"
        ),
        (
            "stale_close_window="
            f"seq{estimates[3].sequence_index} "
            f"mode={estimates[3].mode.value} "
            f"distance={estimates[3].distance_m:.2f}"
        ),
    ]


def audit_pass_on_loss() -> list[str]:
    tracker = GateTracker(
        detector=ScriptedDetector(
            [
                [obs(4.0)],
                [obs(1.30)],
                [],
                [obs(1.05, x=626.0, y=336.0, confidence=0.95)],
            ]
        ),
        camera_params=CAM,
    )

    approach = tracker.update(FRAME, timestamp_s=0.00)
    commit = tracker.update(FRAME, timestamp_s=0.04)
    loss = tracker.update(FRAME, timestamp_s=0.08)
    stale = tracker.update(FRAME, timestamp_s=0.12)

    assert_true(approach.sequence_index == 0, "approach starts at sequence 0")
    assert_true(commit.mode == TrackMode.COMMIT, "near gate should enter commit")
    assert_true(loss.mode == TrackMode.SEARCH, "loss after near commit should enter search")
    assert_true(loss.sequence_index == 1, "loss after near commit should advance sequence")
    assert_true(stale.mode == TrackMode.SEARCH, "post-pass stale close detection should be ignored")
    assert_true(stale.sequence_index == 1, "stale close ignore should keep sequence 1")

    return [
        "scenario=pass_on_detection_loss status=PASS",
        f"loss_after_commit=seq{loss.sequence_index} mode={loss.mode.value}",
        f"stale_reacquire_guard=seq{stale.sequence_index} mode={stale.mode.value}",
    ]


def main() -> int:
    for line in audit_multi_candidate_switch():
        print(line)
    for line in audit_pass_on_loss():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

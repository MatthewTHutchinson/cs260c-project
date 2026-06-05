#!/usr/bin/env python3
"""Smoke-test runtime inference for a learned feature-policy checkpoint."""

from __future__ import annotations

import argparse
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

from algorithm.learned_controller import LearnedFeatureController
from algorithm.types import GateEstimate, TrackMode, VehicleTelemetry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e.pt"),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    controller = LearnedFeatureController(args.checkpoint)
    telemetry = VehicleTelemetry(
        rpy_rad=np.asarray([0.0, -0.08, 0.0], dtype=np.float64),
        linear_velocity_m_s=np.asarray([5.5, 0.15, 0.0], dtype=np.float64),
        timestamp_s=1.0,
    )
    gate = GateEstimate(
        bearing_h_rad=0.12,
        bearing_v_rad=-0.30,
        distance_m=6.0,
        confidence=0.9,
        pixel_center=(358.0, 276.0),
        apparent_size_px=140.0,
        sequence_index=2,
        age_s=0.0,
        mode=TrackMode.DETECTED,
    )

    for _ in range(controller.sequence_length + 2):
        command = controller.compute(gate, telemetry)

    print(f"checkpoint={args.checkpoint}")
    print(f"sequence_length={controller.sequence_length}")
    print(f"roll_rate_rad_s={command.roll_rate_rad_s:.6f}")
    print(f"pitch_rate_rad_s={command.pitch_rate_rad_s:.6f}")
    print(f"yaw_rate_rad_s={command.yaw_rate_rad_s:.6f}")
    print(f"thrust_norm={command.thrust_norm:.6f}")
    print(f"mode={command.mode.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

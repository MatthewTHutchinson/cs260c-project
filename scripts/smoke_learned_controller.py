#!/usr/bin/env python3
"""Smoke-test runtime inference for a learned feature-policy checkpoint."""

from __future__ import annotations

import argparse
import csv
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


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def gate_from_row(row: dict[str, str]) -> GateEstimate:
    mode = row.get("mode", "detected")
    try:
        track_mode = TrackMode(mode)
    except ValueError:
        track_mode = TrackMode.DETECTED
    return GateEstimate(
        bearing_h_rad=as_float(row, "bearing_h_rad"),
        bearing_v_rad=as_float(row, "bearing_v_rad"),
        distance_m=as_float(row, "distance_m"),
        confidence=as_float(row, "confidence"),
        pixel_center=(as_float(row, "pixel_x"), as_float(row, "pixel_y")),
        apparent_size_px=as_float(row, "apparent_size_px"),
        sequence_index=int(as_float(row, "next_gate_index", 0.0)),
        age_s=as_float(row, "gate_age_s"),
        mode=track_mode,
    )


def telemetry_from_row(row: dict[str, str]) -> VehicleTelemetry:
    return VehicleTelemetry(
        rpy_rad=np.asarray([0.0, as_float(row, "body_forward_elevation_rad"), 0.0]),
        linear_velocity_m_s=np.asarray(
            [
                as_float(row, "body_vx_m_s"),
                as_float(row, "body_vy_m_s"),
                as_float(row, "body_vz_m_s"),
            ],
            dtype=np.float64,
        ),
        timestamp_s=as_float(row, "timestamp_s"),
    )


def load_trace_rows(path: Path, course: str | None, limit: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if course:
        rows = [row for row in rows if row.get("course") == course]
    rows = [row for row in rows if row.get("mode") in {TrackMode.DETECTED.value, TrackMode.COMMIT.value}]
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e.pt"),
    )
    parser.add_argument("--trace", type=Path, default=Path("logs/privileged_teacher/trace_with_variants.csv"))
    parser.add_argument("--course", default="s_curve")
    parser.add_argument("--rows", type=int, default=24)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    controller = LearnedFeatureController(args.checkpoint)
    rows = load_trace_rows(args.trace, args.course, args.rows) if args.trace.exists() else []
    if rows:
        for row in rows:
            command = controller.compute(gate_from_row(row), telemetry_from_row(row))
        source = str(args.trace)
        course = args.course
        row_count = len(rows)
    else:
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
        source = "synthetic_single_gate"
        course = ""
        row_count = controller.sequence_length + 2

    print(f"checkpoint={args.checkpoint}")
    print(f"source={source}")
    print(f"course={course}")
    print(f"rows_replayed={row_count}")
    print(f"sequence_length={controller.sequence_length}")
    print(f"roll_rate_rad_s={command.roll_rate_rad_s:.6f}")
    print(f"pitch_rate_rad_s={command.pitch_rate_rad_s:.6f}")
    print(f"yaw_rate_rad_s={command.yaw_rate_rad_s:.6f}")
    print(f"thrust_norm={command.thrust_norm:.6f}")
    print(f"mode={command.mode.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

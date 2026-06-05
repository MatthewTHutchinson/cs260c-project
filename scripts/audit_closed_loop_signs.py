#!/usr/bin/env python3
"""Audit whether commands reduce privileged lateral gate error in a rollout."""

from __future__ import annotations

import argparse
import csv
import math
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

from algorithm.course_library import course_by_name, course_names


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = as_float(row, key, float(default))
    if not math.isfinite(value):
        return default
    return int(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return rows


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-9:
        return float("nan")
    return float((aa @ bb) / denom)


def audit(rows: list[dict[str, str]], course: str) -> dict[str, float]:
    gates = course_by_name(course)
    samples: list[dict[str, float]] = []
    for row in rows:
        if "debug_world_y_m" not in row:
            continue
        next_idx = int(np.clip(as_int(row, "next_gate_index", 0), 0, len(gates) - 1))
        gate = gates[next_idx]
        world_x = as_float(row, "debug_world_x_m")
        world_y = as_float(row, "debug_world_y_m")
        if not math.isfinite(world_x) or not math.isfinite(world_y):
            continue
        lateral_error = world_y - float(gate.center[1])
        samples.append(
            {
                "t": as_float(row, "timestamp_s", 0.0),
                "x": world_x,
                "y": world_y,
                "lateral_error": lateral_error,
                "abs_lateral_error": abs(lateral_error),
                "bearing_h": as_float(row, "bearing_h_rad", 0.0),
                "roll": as_float(row, "roll_rate_rad_s", 0.0),
                "yaw": as_float(row, "yaw_rate_rad_s", 0.0),
                "thrust": as_float(row, "thrust_norm", 0.0),
            }
        )
    if len(samples) < 3:
        raise ValueError("not enough debug-world samples to audit")

    t = np.asarray([s["t"] for s in samples], dtype=np.float64)
    lateral = np.asarray([s["lateral_error"] for s in samples], dtype=np.float64)
    abs_lateral = np.asarray([s["abs_lateral_error"] for s in samples], dtype=np.float64)
    bearing = np.asarray([s["bearing_h"] for s in samples], dtype=np.float64)
    roll = np.asarray([s["roll"] for s in samples], dtype=np.float64)
    yaw = np.asarray([s["yaw"] for s in samples], dtype=np.float64)
    thrust = np.asarray([s["thrust"] for s in samples], dtype=np.float64)

    dt = np.maximum(np.diff(t), 1e-3)
    d_lateral = np.diff(lateral) / dt
    d_abs_lateral = np.diff(abs_lateral) / dt
    roll_mid = roll[:-1]
    yaw_mid = yaw[:-1]
    bearing_mid = bearing[:-1]

    return {
        "samples": float(len(samples)),
        "start_lateral_error_m": float(lateral[0]),
        "end_lateral_error_m": float(lateral[-1]),
        "max_abs_lateral_error_m": float(np.max(abs_lateral)),
        "mean_thrust": float(np.mean(thrust)),
        "corr_roll_to_lateral_rate": corr(roll_mid, d_lateral),
        "corr_yaw_to_lateral_rate": corr(yaw_mid, d_lateral),
        "corr_bearing_to_lateral_rate": corr(bearing_mid, d_lateral),
        "corr_roll_to_abs_error_rate": corr(roll_mid, d_abs_lateral),
        "corr_yaw_to_abs_error_rate": corr(yaw_mid, d_abs_lateral),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--course", choices=course_names(), required=True)
    args = parser.parse_args()

    stats = audit(load_rows(args.trace), args.course)
    print(f"trace={args.trace}")
    print(f"course={args.course}")
    for key, value in stats.items():
        if key == "samples":
            print(f"{key}={int(value)}")
        else:
            print(f"{key}={value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

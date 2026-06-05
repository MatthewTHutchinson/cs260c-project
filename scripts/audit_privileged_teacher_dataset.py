#!/usr/bin/env python3
"""Audit privileged-teacher datasets before spending training time."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
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

import matplotlib.pyplot as plt
import numpy as np


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"dataset has no rows: {path}")
    return rows


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, q))


def summarize_course(course: str, rows: list[dict[str, str]]) -> list[str]:
    bearing_h = np.asarray([as_float(row, "bearing_h_rad") for row in rows], dtype=np.float64)
    distance = np.asarray([as_float(row, "distance_m") for row in rows], dtype=np.float64)
    yaw = np.asarray([as_float(row, "teacher_yaw_rate_rad_s") for row in rows], dtype=np.float64)
    roll = np.asarray([as_float(row, "teacher_roll_rate_rad_s") for row in rows], dtype=np.float64)
    pitch = np.asarray([as_float(row, "teacher_pitch_rate_rad_s") for row in rows], dtype=np.float64)
    thrust = np.asarray([as_float(row, "teacher_thrust_norm") for row in rows], dtype=np.float64)
    target_y = np.asarray([as_float(row, "teacher_target_y_m") for row in rows], dtype=np.float64)
    world_y = np.asarray([as_float(row, "world_y_m") for row in rows], dtype=np.float64)
    lateral_target_delta = target_y - world_y

    return [
        (
            f"course={course} rows={len(rows)} "
            f"abs_bearing_h_mean={np.nanmean(np.abs(bearing_h)):.4f} "
            f"abs_bearing_h_p95={percentile(np.abs(bearing_h), 95):.4f} "
            f"distance_min={np.nanmin(distance):.2f} distance_max={np.nanmax(distance):.2f}"
        ),
        (
            f"  teacher_cmd_ranges "
            f"roll=[{np.nanmin(roll):.3f},{np.nanmax(roll):.3f}] "
            f"pitch=[{np.nanmin(pitch):.3f},{np.nanmax(pitch):.3f}] "
            f"yaw=[{np.nanmin(yaw):.3f},{np.nanmax(yaw):.3f}] "
            f"thrust=[{np.nanmin(thrust):.3f},{np.nanmax(thrust):.3f}]"
        ),
        (
            f"  lateral_lookahead_delta "
            f"mean_abs={np.nanmean(np.abs(lateral_target_delta)):.3f} "
            f"p95_abs={percentile(np.abs(lateral_target_delta), 95):.3f}"
        ),
    ]


def group_by_course(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("course", "unknown")].append(row)
    return dict(sorted(grouped.items()))


def plot_courses(grouped: dict[str, list[dict[str, str]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for course, rows in grouped.items():
        t = np.asarray([as_float(row, "timestamp_s") for row in rows], dtype=np.float64)
        world_x = np.asarray([as_float(row, "world_x_m") for row in rows], dtype=np.float64)
        world_y = np.asarray([as_float(row, "world_y_m") for row in rows], dtype=np.float64)
        target_x = np.asarray([as_float(row, "teacher_target_x_m") for row in rows], dtype=np.float64)
        target_y = np.asarray([as_float(row, "teacher_target_y_m") for row in rows], dtype=np.float64)
        gate_x = np.asarray([as_float(row, "teacher_next_gate_x_m") for row in rows], dtype=np.float64)
        gate_y = np.asarray([as_float(row, "teacher_next_gate_y_m") for row in rows], dtype=np.float64)
        roll = np.asarray([as_float(row, "teacher_roll_rate_rad_s") for row in rows], dtype=np.float64)
        pitch = np.asarray([as_float(row, "teacher_pitch_rate_rad_s") for row in rows], dtype=np.float64)
        yaw = np.asarray([as_float(row, "teacher_yaw_rate_rad_s") for row in rows], dtype=np.float64)
        thrust = np.asarray([as_float(row, "teacher_thrust_norm") for row in rows], dtype=np.float64)
        bearing_h = np.asarray([as_float(row, "bearing_h_rad") for row in rows], dtype=np.float64)

        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0))
        fig.suptitle(f"Privileged Teacher Audit: {course}", fontsize=14, fontweight="bold")

        axes[0].plot(world_x, world_y, color="#1971c2", linewidth=2.0, label="teacher path")
        axes[0].plot(target_x, target_y, color="#2f9e44", linewidth=1.2, alpha=0.8, label="lookahead target")
        axes[0].scatter(gate_x, gate_y, color="#e67700", s=18, label="next gate center")
        axes[0].set_ylabel("world y (m)")
        axes[0].set_xlabel("world x (m)")
        axes[0].axis("equal")
        axes[0].grid(True, color="#d9dde3")
        axes[0].legend(frameon=False, loc="best")

        axes[1].plot(t, bearing_h, color="#e67700", linewidth=1.5, label="bearing_h")
        axes[1].axhline(0.0, color="#868e96", linewidth=1.0)
        axes[1].set_ylabel("bearing rad")
        axes[1].grid(True, color="#d9dde3")
        axes[1].legend(frameon=False, loc="best")

        axes[2].plot(t, roll, label="roll", color="#5f3dc4", linewidth=1.4)
        axes[2].plot(t, pitch, label="pitch", color="#c2255c", linewidth=1.4)
        axes[2].plot(t, yaw, label="yaw", color="#1971c2", linewidth=1.4)
        axes[2].plot(t, thrust, label="thrust", color="#2b8a3e", linewidth=1.4)
        axes[2].set_ylabel("teacher command")
        axes[2].set_xlabel("time (s)")
        axes[2].grid(True, color="#d9dde3")
        axes[2].legend(frameon=False, loc="best", ncol=4)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"{course}_teacher_audit.png", dpi=170)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/privileged_teacher/trace.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/privileged_teacher/audit"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.trace)
    grouped = group_by_course(rows)
    print(f"trace={args.trace}")
    print(f"rows={len(rows)}")
    print(f"courses={','.join(grouped)}")
    for course, course_rows in grouped.items():
        for line in summarize_course(course, course_rows):
            print(line)
    if args.plot:
        plot_courses(grouped, args.out_dir)
        print(f"plots={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


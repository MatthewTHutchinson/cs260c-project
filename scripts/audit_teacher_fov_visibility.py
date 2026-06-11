#!/usr/bin/env python3
"""Audit teacher-track gate/lookahead visibility under camera profiles."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from algorithm.camera_profiles import get_camera_profile


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return rows


def yaw_to_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.asarray(
        (
            (c, -s, 0.0),
            (s, c, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def camera_bearing_to_point(
    row: dict[str, str],
    *,
    x_key: str,
    y_key: str,
    z_key: str,
    camera_tilt_up_rad: float,
) -> tuple[float, float, float]:
    position = np.asarray(
        (
            as_float(row, "world_x_m"),
            as_float(row, "world_y_m"),
            as_float(row, "world_z_m"),
        ),
        dtype=np.float64,
    )
    target = np.asarray(
        (
            as_float(row, x_key),
            as_float(row, y_key),
            as_float(row, z_key),
        ),
        dtype=np.float64,
    )
    rel_body = yaw_to_matrix(as_float(row, "world_yaw_rad")).T @ (target - position)
    forward = float(rel_body[0])
    lateral = float(rel_body[1])
    vertical = float(rel_body[2])
    distance = float(np.linalg.norm(rel_body))
    if forward <= 1e-6:
        return math.copysign(math.pi / 2.0, lateral if lateral else 1.0), math.pi / 2.0, distance
    bearing_h = math.atan2(lateral, forward)
    bearing_v_body = math.atan2(vertical, max(1e-6, math.hypot(forward, lateral)))
    return bearing_h, bearing_v_body - camera_tilt_up_rad, distance


def is_visible(bearing_h: float, bearing_v: float, profile_name: str) -> bool:
    profile = get_camera_profile(profile_name)
    half_h = math.radians(profile.fov_horiz_deg / 2.0)
    half_v = math.radians(profile.fov_vert_deg / 2.0)
    return abs(bearing_h) <= half_h and abs(bearing_v) <= half_v


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("course", "unknown") or "unknown"].append(row)
    return dict(sorted(grouped.items()))


def summarize_visibility(
    rows: list[dict[str, str]],
    *,
    profile_name: str,
) -> dict[str, float]:
    profile = get_camera_profile(profile_name)
    next_visible: list[bool] = []
    lookahead_visible: list[bool] = []
    abs_next_h: list[float] = []
    abs_lookahead_h: list[float] = []
    for row in rows:
        next_h = as_float(row, "bearing_h_rad")
        next_v = as_float(row, "bearing_v_rad")
        look_h, look_v, _look_dist = camera_bearing_to_point(
            row,
            x_key="teacher_target_x_m",
            y_key="teacher_target_y_m",
            z_key="teacher_target_z_m",
            camera_tilt_up_rad=math.radians(profile.tilt_up_deg),
        )
        next_visible.append(is_visible(next_h, next_v, profile_name))
        lookahead_visible.append(is_visible(look_h, look_v, profile_name))
        abs_next_h.append(abs(next_h))
        abs_lookahead_h.append(abs(look_h))
    return {
        "rows": float(len(rows)),
        "next_gate_visible_pct": float(np.mean(next_visible) * 100.0) if next_visible else 0.0,
        "lookahead_visible_pct": float(np.mean(lookahead_visible) * 100.0) if lookahead_visible else 0.0,
        "next_gate_abs_bearing_h_p95_deg": float(np.degrees(np.percentile(abs_next_h, 95))) if abs_next_h else 0.0,
        "lookahead_abs_bearing_h_p95_deg": float(np.degrees(np.percentile(abs_lookahead_h, 95))) if abs_lookahead_h else 0.0,
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "profile",
        "course",
        "rows",
        "next_gate_visible_pct",
        "lookahead_visible_pct",
        "next_gate_abs_bearing_h_p95_deg",
        "lookahead_abs_bearing_h_p95_deg",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/privileged_teacher/trace_augmented_rejoin.csv"))
    parser.add_argument("--profiles", default="vq1_pinhole,gatenet_fisheye")
    parser.add_argument("--phases", default="nominal,off_nominal")
    parser.add_argument("--out", type=Path, default=Path("logs/privileged_teacher/fov_visibility_summary.csv"))
    parser.add_argument("--warn-visible-pct", type=float, default=90.0)
    args = parser.parse_args()

    selected_profiles = split_csv(args.profiles)
    selected_phases = set(split_csv(args.phases))
    rows = [
        row
        for row in load_rows(args.trace)
        if not selected_phases or row.get("teacher_phase", "") in selected_phases
    ]
    grouped = group_rows(rows)
    summary_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    print(f"trace={args.trace}")
    print(f"phases={','.join(sorted(selected_phases)) if selected_phases else 'all'}")
    for profile_name in selected_profiles:
        profile = get_camera_profile(profile_name)
        print(
            f"profile={profile.name} "
            f"fov_h={profile.fov_horiz_deg:.1f} "
            f"fov_v={profile.fov_vert_deg:.1f} "
            f"tilt_up={profile.tilt_up_deg:.1f}"
        )
        all_stats = summarize_visibility(rows, profile_name=profile_name)
        print(
            f"  overall rows={int(all_stats['rows'])} "
            f"next_gate_visible={all_stats['next_gate_visible_pct']:.1f}% "
            f"lookahead_visible={all_stats['lookahead_visible_pct']:.1f}% "
            f"next_h_p95={all_stats['next_gate_abs_bearing_h_p95_deg']:.1f}deg "
            f"lookahead_h_p95={all_stats['lookahead_abs_bearing_h_p95_deg']:.1f}deg"
        )
        for course, course_rows in grouped.items():
            stats = summarize_visibility(course_rows, profile_name=profile_name)
            summary_rows.append({"profile": profile_name, "course": course, **stats})
            if stats["next_gate_visible_pct"] < args.warn_visible_pct:
                warnings.append(
                    f"profile={profile_name} course={course} "
                    f"next_gate_visible_pct={stats['next_gate_visible_pct']:.1f}"
                )

    write_summary(args.out, summary_rows)
    print(f"summary={args.out}")
    if warnings:
        print("warnings:")
        for warning in warnings[:20]:
            print(f"  {warning}")
        if len(warnings) > 20:
            print(f"  additional_warnings={len(warnings) - 20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate privileged-teacher labels from known debug course geometry.

The output CSV deliberately separates legal student features from privileged
teacher/debug fields. Use this for BC data plumbing and teacher-shape iteration,
not as evidence that the deployed policy consumes gate truth.
"""

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

from algorithm.course_library import CourseGate, course_by_name, course_names


FIELDNAMES = (
    "timestamp_s",
    "course",
    "mode",
    "confidence",
    "bearing_h_rad",
    "bearing_v_rad",
    "distance_m",
    "pixel_x",
    "pixel_y",
    "apparent_size_px",
    "gate_age_s",
    "body_forward_elevation_rad",
    "body_vx_m_s",
    "body_vy_m_s",
    "body_vz_m_s",
    "prev_roll_rate_rad_s",
    "prev_pitch_rate_rad_s",
    "prev_yaw_rate_rad_s",
    "prev_thrust_norm",
    "roll_rate_rad_s",
    "pitch_rate_rad_s",
    "yaw_rate_rad_s",
    "thrust_norm",
    "teacher_roll_rate_rad_s",
    "teacher_pitch_rate_rad_s",
    "teacher_yaw_rate_rad_s",
    "teacher_thrust_norm",
    "teacher_target_x_m",
    "teacher_target_y_m",
    "teacher_target_z_m",
    "teacher_next_gate_index",
    "teacher_next_gate_x_m",
    "teacher_next_gate_y_m",
    "teacher_next_gate_z_m",
    "teacher_next_gate_yaw_deg",
    "world_x_m",
    "world_y_m",
    "world_z_m",
    "world_yaw_rad",
)


def wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def interp_angle(a: float, b: float, u: float) -> float:
    return float(a + wrap_pi(b - a) * u)


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def yaw_to_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def bearing_to_gate(
    *,
    position: np.ndarray,
    yaw_rad: float,
    gate: CourseGate,
    camera_tilt_up_rad: float,
) -> tuple[float, float, float, float, float]:
    rel_world = gate.center - position
    rel_body = yaw_to_matrix(yaw_rad).T @ rel_world
    forward = max(1e-3, float(rel_body[0]))
    lateral = float(rel_body[1])
    vertical = float(rel_body[2])
    distance = float(np.linalg.norm(rel_body))
    bearing_h = math.atan2(lateral, forward)
    bearing_v_body = math.atan2(vertical, max(1e-3, math.hypot(forward, lateral)))
    bearing_v_camera = bearing_v_body - camera_tilt_up_rad
    return bearing_h, bearing_v_camera, distance, lateral, vertical


def generate_rows(
    *,
    course_name: str,
    gates: tuple[CourseGate, ...],
    samples_per_segment: int,
    speed_m_s: float,
    lookahead_m: float,
    camera_tilt_up_rad: float,
) -> list[dict[str, str]]:
    start = np.asarray([0.0, 0.0, 1.8], dtype=np.float64)
    points = [start, *[gate.center for gate in gates]]
    headings = []
    for i, point in enumerate(points):
        if i < len(points) - 1:
            direction = points[i + 1] - point
        else:
            direction = point - points[i - 1]
        headings.append(math.atan2(float(direction[1]), float(direction[0])))

    rows: list[dict[str, str]] = []
    t = 0.0
    prev_command = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float64)
    for seg_idx in range(len(points) - 1):
        p0 = points[seg_idx]
        p1 = points[seg_idx + 1]
        gate = gates[min(seg_idx, len(gates) - 1)]
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        duration = max(0.2, length / max(speed_m_s, 1e-3))
        dt = duration / samples_per_segment
        for j in range(samples_per_segment):
            u = j / samples_per_segment
            su = smoothstep(u)
            position = p0 + segment * su
            yaw = interp_angle(headings[seg_idx], headings[min(seg_idx + 1, len(headings) - 1)], su)
            tangent = segment / max(length, 1e-6)
            speed_scale = 6.0 * u * (1.0 - u)
            velocity_world = tangent * speed_m_s * max(0.20, speed_scale)
            bearing_h, bearing_v, distance, lateral, vertical = bearing_to_gate(
                position=position,
                yaw_rad=yaw,
                gate=gate,
                camera_tilt_up_rad=camera_tilt_up_rad,
            )

            lookahead_target = position + tangent * min(lookahead_m, max(0.0, distance))
            lookahead_target = 0.45 * gate.center + 0.55 * lookahead_target
            target_body = yaw_to_matrix(yaw).T @ (lookahead_target - position)
            target_yaw = math.atan2(float(target_body[1]), max(1e-3, float(target_body[0])))
            target_pitch = -0.18 - 0.035 * max(0.0, distance - 2.0)
            target_pitch += 0.035 * float(target_body[2])
            target_roll = 0.23 * float(target_body[1]) / max(2.0, distance)
            target_thrust = 0.56 + 0.035 * float(target_body[2])
            command = np.asarray(
                [
                    np.clip(target_roll, -0.70, 0.70),
                    np.clip(target_pitch, -0.80, 0.35),
                    np.clip(1.8 * target_yaw, -1.20, 1.20),
                    np.clip(target_thrust, 0.35, 0.90),
                ],
                dtype=np.float64,
            )
            confidence = float(np.clip(1.0 - abs(bearing_h) / 1.0, 0.15, 0.98))
            apparent_size = 2.7 * 320.0 / max(distance, 1e-3)
            rows.append(
                {
                    "timestamp_s": f"{t:.6f}",
                    "course": course_name,
                    "mode": "commit" if distance < 2.2 else "detected",
                    "confidence": f"{confidence:.6f}",
                    "bearing_h_rad": f"{bearing_h:.6f}",
                    "bearing_v_rad": f"{bearing_v:.6f}",
                    "distance_m": f"{distance:.6f}",
                    "pixel_x": f"{320.0 + math.tan(bearing_h) * 320.0:.3f}",
                    "pixel_y": f"{180.0 - math.tan(bearing_v) * 320.0:.3f}",
                    "apparent_size_px": f"{apparent_size:.3f}",
                    "gate_age_s": "0.000000",
                    "body_forward_elevation_rad": "0.000000",
                    "body_vx_m_s": f"{velocity_world[0]:.6f}",
                    "body_vy_m_s": f"{velocity_world[1]:.6f}",
                    "body_vz_m_s": f"{velocity_world[2]:.6f}",
                    "prev_roll_rate_rad_s": f"{prev_command[0]:.6f}",
                    "prev_pitch_rate_rad_s": f"{prev_command[1]:.6f}",
                    "prev_yaw_rate_rad_s": f"{prev_command[2]:.6f}",
                    "prev_thrust_norm": f"{prev_command[3]:.6f}",
                    "roll_rate_rad_s": f"{command[0]:.6f}",
                    "pitch_rate_rad_s": f"{command[1]:.6f}",
                    "yaw_rate_rad_s": f"{command[2]:.6f}",
                    "thrust_norm": f"{command[3]:.6f}",
                    "teacher_roll_rate_rad_s": f"{command[0]:.6f}",
                    "teacher_pitch_rate_rad_s": f"{command[1]:.6f}",
                    "teacher_yaw_rate_rad_s": f"{command[2]:.6f}",
                    "teacher_thrust_norm": f"{command[3]:.6f}",
                    "teacher_target_x_m": f"{lookahead_target[0]:.6f}",
                    "teacher_target_y_m": f"{lookahead_target[1]:.6f}",
                    "teacher_target_z_m": f"{lookahead_target[2]:.6f}",
                    "teacher_next_gate_index": str(gate.index),
                    "teacher_next_gate_x_m": f"{gate.center[0]:.6f}",
                    "teacher_next_gate_y_m": f"{gate.center[1]:.6f}",
                    "teacher_next_gate_z_m": f"{gate.center[2]:.6f}",
                    "teacher_next_gate_yaw_deg": f"{gate.yaw_deg:.6f}",
                    "world_x_m": f"{position[0]:.6f}",
                    "world_y_m": f"{position[1]:.6f}",
                    "world_z_m": f"{position[2]:.6f}",
                    "world_yaw_rad": f"{yaw:.6f}",
                }
            )
            prev_command = command
            t += dt
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--courses",
        default="easy,lateral_soft,low_high,four_gate_straight,circular_arc,s_curve",
        help=f"Comma-separated course names. Available: {', '.join(course_names())}",
    )
    parser.add_argument("--out", type=Path, default=Path("logs/privileged_teacher/trace.csv"))
    parser.add_argument("--samples-per-segment", type=int, default=90)
    parser.add_argument("--speed-m-s", type=float, default=6.0)
    parser.add_argument("--lookahead-m", type=float, default=5.0)
    parser.add_argument("--camera-tilt-up-deg", type=float, default=20.0)
    args = parser.parse_args()

    all_rows: list[dict[str, str]] = []
    for course_name in [c.strip() for c in args.courses.split(",") if c.strip()]:
        all_rows.extend(
            generate_rows(
                course_name=course_name,
                gates=course_by_name(course_name),
                samples_per_segment=args.samples_per_segment,
                speed_m_s=args.speed_m_s,
                lookahead_m=args.lookahead_m,
                camera_tilt_up_rad=math.radians(args.camera_tilt_up_deg),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"dataset={args.out}")
    print(f"rows={len(all_rows)}")
    print("student_inputs=FPV-derived features, tracker-like mode/history, telemetry")
    print("privileged_fields=teacher_* and world_* columns; do not feed to deployed policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
    "teacher_heading_rate_rad_s",
    "teacher_world_vx_m_s",
    "teacher_world_vy_m_s",
    "teacher_world_vz_m_s",
    "teacher_world_ax_m_s2",
    "teacher_world_ay_m_s2",
    "teacher_world_az_m_s2",
    "world_x_m",
    "world_y_m",
    "world_z_m",
    "world_yaw_rad",
)


def hermite_blend(u: float) -> tuple[float, float, float, float]:
    """Cubic Hermite basis for a smooth through-gate racing reference."""
    u2 = u * u
    u3 = u2 * u
    return (
        2.0 * u3 - 3.0 * u2 + 1.0,
        u3 - 2.0 * u2 + u,
        -2.0 * u3 + 3.0 * u2,
        u3 - u2,
    )


def hermite_blend_derivative(u: float) -> tuple[float, float, float, float]:
    u2 = u * u
    return (
        6.0 * u2 - 6.0 * u,
        3.0 * u2 - 4.0 * u + 1.0,
        -6.0 * u2 + 6.0 * u,
        3.0 * u2 - 2.0 * u,
    )


def hermite_blend_second_derivative(u: float) -> tuple[float, float, float, float]:
    return (
        12.0 * u - 6.0,
        6.0 * u - 4.0,
        -12.0 * u + 6.0,
        6.0 * u - 2.0,
    )


def tangent_at(points: list[np.ndarray], index: int, speed_m_s: float) -> np.ndarray:
    if index == 0:
        direction = points[1] - points[0]
    elif index == len(points) - 1:
        direction = points[-1] - points[-2]
    else:
        direction = points[index + 1] - points[index - 1]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.asarray([speed_m_s, 0.0, 0.0], dtype=np.float64)
    return direction / norm * speed_m_s


def hermite_state(
    *,
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    u: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h00, h10, h01, h11 = hermite_blend(u)
    dh00, dh10, dh01, dh11 = hermite_blend_derivative(u)
    ddh00, ddh10, ddh01, ddh11 = hermite_blend_second_derivative(u)
    m0 = v0 * duration_s
    m1 = v1 * duration_s
    position = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    velocity = (dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1) / duration_s
    acceleration = (
        ddh00 * p0 + ddh10 * m0 + ddh01 * p1 + ddh11 * m1
    ) / max(duration_s * duration_s, 1e-6)
    return position, velocity, acceleration


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
    tangents = [tangent_at(points, i, speed_m_s) for i in range(len(points))]

    samples: list[dict[str, object]] = []
    t = 0.0
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
            position, velocity_world, acceleration_world = hermite_state(
                p0=p0,
                p1=p1,
                v0=tangents[seg_idx],
                v1=tangents[seg_idx + 1],
                u=u,
                duration_s=duration,
            )
            yaw = math.atan2(float(velocity_world[1]), max(1e-6, float(velocity_world[0])))
            speed_xy_sq = max(1e-6, float(velocity_world[0] ** 2 + velocity_world[1] ** 2))
            heading_rate = float(
                (
                    velocity_world[0] * acceleration_world[1]
                    - velocity_world[1] * acceleration_world[0]
                )
                / speed_xy_sq
            )
            samples.append(
                {
                    "timestamp_s": t,
                    "gate": gate,
                    "position": position,
                    "velocity_world": velocity_world,
                    "acceleration_world": acceleration_world,
                    "yaw": yaw,
                    "heading_rate": heading_rate,
                }
            )
            t += dt

    if not samples:
        return []

    cumulative_distance = [0.0]
    for prev, cur in zip(samples, samples[1:]):
        prev_position = prev["position"]
        cur_position = cur["position"]
        assert isinstance(prev_position, np.ndarray)
        assert isinstance(cur_position, np.ndarray)
        cumulative_distance.append(
            cumulative_distance[-1] + float(np.linalg.norm(cur_position - prev_position))
        )
    cumulative_distance_np = np.asarray(cumulative_distance, dtype=np.float64)

    rows: list[dict[str, str]] = []
    prev_command = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float64)
    for sample_idx, sample in enumerate(samples):
        gate = sample["gate"]
        position = sample["position"]
        velocity_world = sample["velocity_world"]
        acceleration_world = sample["acceleration_world"]
        yaw = sample["yaw"]
        heading_rate = sample["heading_rate"]
        t = sample["timestamp_s"]
        assert isinstance(gate, CourseGate)
        assert isinstance(position, np.ndarray)
        assert isinstance(velocity_world, np.ndarray)
        assert isinstance(acceleration_world, np.ndarray)
        assert isinstance(yaw, float)
        assert isinstance(heading_rate, float)
        assert isinstance(t, float)

        lookahead_distance = cumulative_distance_np[sample_idx] + lookahead_m
        lookahead_idx = int(np.searchsorted(cumulative_distance_np, lookahead_distance, side="left"))
        lookahead_idx = min(lookahead_idx, len(samples) - 1)
        lookahead_target = samples[lookahead_idx]["position"]
        assert isinstance(lookahead_target, np.ndarray)

        bearing_h, bearing_v, distance, _lateral, _vertical = bearing_to_gate(
            position=position,
            yaw_rad=yaw,
            gate=gate,
            camera_tilt_up_rad=camera_tilt_up_rad,
        )
        observed_distance = max(distance, 0.50)

        target_body = yaw_to_matrix(yaw).T @ (lookahead_target - position)
        velocity_body = yaw_to_matrix(yaw).T @ velocity_world
        acceleration_body = yaw_to_matrix(yaw).T @ acceleration_world
        target_yaw = math.atan2(float(target_body[1]), max(1e-3, float(target_body[0])))
        target_pitch = -0.16 - 0.040 * max(0.0, float(velocity_body[0]) - 3.0)
        target_pitch += 0.035 * float(target_body[2])
        target_roll = 0.11 * float(acceleration_body[1]) + 0.18 * float(target_body[1]) / max(2.0, distance)
        target_thrust = 0.56 + 0.020 * float(acceleration_body[2]) + 0.030 * float(target_body[2])
        command = np.asarray(
            [
                np.clip(target_roll, -0.70, 0.70),
                np.clip(target_pitch, -0.80, 0.35),
                np.clip(heading_rate + 1.25 * target_yaw, -1.20, 1.20),
                np.clip(target_thrust, 0.35, 0.90),
            ],
            dtype=np.float64,
        )
        confidence = float(np.clip(1.0 - abs(bearing_h) / 1.0, 0.15, 0.98))
        apparent_size = 2.7 * 320.0 / observed_distance
        rows.append(
            {
                "timestamp_s": f"{t:.6f}",
                "course": course_name,
                "mode": "commit" if distance < 2.2 else "detected",
                "confidence": f"{confidence:.6f}",
                "bearing_h_rad": f"{bearing_h:.6f}",
                "bearing_v_rad": f"{bearing_v:.6f}",
                "distance_m": f"{observed_distance:.6f}",
                "pixel_x": f"{320.0 + math.tan(bearing_h) * 320.0:.3f}",
                "pixel_y": f"{180.0 - math.tan(bearing_v) * 320.0:.3f}",
                "apparent_size_px": f"{apparent_size:.3f}",
                "gate_age_s": "0.000000",
                "body_forward_elevation_rad": "0.000000",
                "body_vx_m_s": f"{velocity_body[0]:.6f}",
                "body_vy_m_s": f"{velocity_body[1]:.6f}",
                "body_vz_m_s": f"{velocity_body[2]:.6f}",
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
                "teacher_heading_rate_rad_s": f"{heading_rate:.6f}",
                "teacher_world_vx_m_s": f"{velocity_world[0]:.6f}",
                "teacher_world_vy_m_s": f"{velocity_world[1]:.6f}",
                "teacher_world_vz_m_s": f"{velocity_world[2]:.6f}",
                "teacher_world_ax_m_s2": f"{acceleration_world[0]:.6f}",
                "teacher_world_ay_m_s2": f"{acceleration_world[1]:.6f}",
                "teacher_world_az_m_s2": f"{acceleration_world[2]:.6f}",
                "world_x_m": f"{position[0]:.6f}",
                "world_y_m": f"{position[1]:.6f}",
                "world_z_m": f"{position[2]:.6f}",
                "world_yaw_rad": f"{yaw:.6f}",
            }
        )
        prev_command = command
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

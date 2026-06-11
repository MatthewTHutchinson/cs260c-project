#!/usr/bin/env python3
"""Relabel a closed-loop Elodin trace with privileged teacher commands.

The input trace comes from the deployed pilot and contains legal runtime
features plus debug-only world pose columns. The output is a BC-compatible
teacher CSV: student inputs remain FPV/tracker/telemetry fields, while the new
teacher targets are computed offline from privileged course geometry.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from scripts.generate_privileged_teacher_dataset import (
    FIELDNAMES,
    bearing_to_gate,
    clamp_command,
    hermite_state,
    recovery_command_from_state,
    tangent_at,
    yaw_to_matrix,
)


USABLE_MODES = {"detected", "tracked", "commit"}


@dataclass(frozen=True)
class ReferenceSample:
    position: np.ndarray
    velocity_world: np.ndarray
    gate: CourseGate
    cumulative_distance_m: float


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


def split_csv(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    missing = [name for name in ("debug_world_x_m", "debug_world_y_m", "debug_world_z_m", "debug_world_yaw_rad") if name not in rows[0]]
    if missing:
        raise ValueError(
            f"trace is missing debug-world columns {missing}; rerun Elodin after applying the latest patch"
        )
    return rows


def finite_velocity_world(row: dict[str, str], yaw_rad: float) -> np.ndarray:
    body_velocity = np.asarray(
        [
            as_float(row, "body_vx_m_s", 0.0),
            as_float(row, "body_vy_m_s", 0.0),
            as_float(row, "body_vz_m_s", 0.0),
        ],
        dtype=np.float64,
    )
    return yaw_to_matrix(yaw_rad) @ body_velocity


def choose_gate(row: dict[str, str], gates: tuple[CourseGate, ...]) -> CourseGate:
    next_idx = as_int(row, "next_gate_index", as_int(row, "last_gate_passed", -1) + 1)
    next_idx = int(np.clip(next_idx, 0, len(gates) - 1))
    return gates[next_idx]


def teacher_target(gate: CourseGate, lookahead_m: float) -> np.ndarray:
    return gate.center + gate.normal_enu * lookahead_m


def build_reference_samples(
    gates: tuple[CourseGate, ...],
    *,
    speed_m_s: float,
    samples_per_segment: int,
) -> list[ReferenceSample]:
    start = np.asarray([0.0, 0.0, 1.8], dtype=np.float64)
    points = [start, *[gate.center for gate in gates]]
    tangents = [tangent_at(points, i, speed_m_s) for i in range(len(points))]
    samples: list[ReferenceSample] = []
    cumulative_distance = 0.0
    last_position: np.ndarray | None = None

    for seg_idx in range(len(points) - 1):
        p0 = points[seg_idx]
        p1 = points[seg_idx + 1]
        gate = gates[min(seg_idx, len(gates) - 1)]
        segment_length = float(np.linalg.norm(p1 - p0))
        duration_s = max(0.2, segment_length / max(speed_m_s, 1e-3))
        for j in range(samples_per_segment):
            u = j / samples_per_segment
            position, velocity_world, _acceleration_world = hermite_state(
                p0=p0,
                p1=p1,
                v0=tangents[seg_idx],
                v1=tangents[seg_idx + 1],
                u=u,
                duration_s=duration_s,
            )
            if last_position is not None:
                cumulative_distance += float(np.linalg.norm(position - last_position))
            samples.append(
                ReferenceSample(
                    position=position,
                    velocity_world=velocity_world,
                    gate=gate,
                    cumulative_distance_m=cumulative_distance,
                )
            )
            last_position = position

    if last_position is not None:
        final_gate = gates[-1]
        final_position = final_gate.center
        cumulative_distance += float(np.linalg.norm(final_position - last_position))
        samples.append(
            ReferenceSample(
                position=final_position,
                velocity_world=tangents[-1],
                gate=final_gate,
                cumulative_distance_m=cumulative_distance,
            )
        )
    return samples


def nearest_reference_index(
    reference_samples: list[ReferenceSample],
    *,
    position: np.ndarray,
    gate_index: int,
) -> int:
    candidate_indices = [
        idx for idx, sample in enumerate(reference_samples) if sample.gate.index == gate_index
    ]
    if not candidate_indices:
        candidate_indices = list(range(len(reference_samples)))
    best_idx = candidate_indices[0]
    best_dist = float("inf")
    for idx in candidate_indices:
        sample = reference_samples[idx]
        delta = sample.position - position
        dist = float(np.linalg.norm(delta))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def lookahead_reference(
    reference_samples: list[ReferenceSample],
    *,
    start_idx: int,
    lookahead_m: float,
) -> ReferenceSample:
    target_distance = reference_samples[start_idx].cumulative_distance_m + lookahead_m
    lo = start_idx
    hi = len(reference_samples) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reference_samples[mid].cumulative_distance_m < target_distance:
            lo = mid + 1
        else:
            hi = mid
    return reference_samples[lo]


def relabel_command(
    *,
    position: np.ndarray,
    yaw_rad: float,
    velocity_world: np.ndarray,
    gate: CourseGate,
    lookahead_m: float,
    observed_bearing_h: float,
    teacher: str,
    reference_position: np.ndarray | None = None,
    reference_velocity_world: np.ndarray | None = None,
    rejoin_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    target = teacher_target(gate, lookahead_m)
    bearing_h, _bearing_v, distance, _lateral, _vertical = bearing_to_gate(
        position=position,
        yaw_rad=yaw_rad,
        gate=gate,
        camera_tilt_up_rad=math.radians(20.0),
    )
    if teacher == "rejoin":
        if (
            reference_position is None
            or reference_velocity_world is None
            or rejoin_target is None
        ):
            raise ValueError("rejoin relabeling requires reference state and target")
        command = recovery_command_from_state(
            position=position,
            yaw=yaw_rad,
            velocity_world=velocity_world,
            reference_position=reference_position,
            reference_velocity_world=reference_velocity_world,
            rejoin_target=rejoin_target,
        )
        return command, rejoin_target, distance

    target_body = yaw_to_matrix(yaw_rad).T @ (target - position)
    velocity_body = yaw_to_matrix(yaw_rad).T @ velocity_world
    heading_error = math.atan2(float(target_body[1]), max(1.0, float(target_body[0])))
    altitude_error = float(gate.center[2] - position[2])

    # A relabeling teacher should recover missed approaches, not merely imitate
    # the failed rollout. These gains intentionally emphasize lateral/yaw
    # correction and altitude hold while still using the same command boundary.
    # The Elodin/Betaflight adapter sign follows visual bearing: positive
    # bearing needs positive roll/yaw to recenter the gate in this harness.
    visual_error = observed_bearing_h if math.isfinite(observed_bearing_h) else bearing_h
    roll = 0.52 * visual_error - 0.05 * float(velocity_body[1])
    pitch = -0.14 - 0.035 * max(0.0, float(velocity_body[0]) - 3.0)
    if abs(heading_error) > 0.30:
        pitch += 0.07
    if abs(visual_error) > 0.25:
        pitch = max(pitch, -0.04)
    yaw = 1.55 * visual_error
    thrust = 0.60 + 0.10 * altitude_error - 0.045 * float(velocity_world[2])
    thrust = max(thrust, 0.66 if abs(visual_error) > 0.18 else 0.60)
    command = clamp_command((roll, pitch, yaw, thrust))
    return command, target, distance


def relabel_rows(
    rows: list[dict[str, str]],
    *,
    course_name: str,
    episode_id: str,
    lookahead_m: float,
    min_confidence: float,
    max_past_gate_m: float,
    max_reference_error_m: float,
    allowed_state_command_sources: set[str] | None,
    teacher: str,
    reference_speed_m_s: float,
    reference_samples_per_segment: int,
) -> list[dict[str, str]]:
    gates = course_by_name(course_name)
    reference_samples = build_reference_samples(
        gates,
        speed_m_s=reference_speed_m_s,
        samples_per_segment=reference_samples_per_segment,
    )
    out: list[dict[str, str]] = []
    prev_command = np.asarray([0.0, 0.0, 0.0, 0.55], dtype=np.float64)
    last_velocity: np.ndarray | None = None
    last_t: float | None = None

    for row in rows:
        mode = row.get("mode", "").strip().lower()
        if mode not in USABLE_MODES:
            continue
        command_source = row.get("command_source", "").strip()
        if (
            allowed_state_command_sources is not None
            and command_source not in allowed_state_command_sources
        ):
            continue
        if as_float(row, "confidence", 0.0) < min_confidence:
            continue
        position = np.asarray(
            [
                as_float(row, "debug_world_x_m"),
                as_float(row, "debug_world_y_m"),
                as_float(row, "debug_world_z_m"),
            ],
            dtype=np.float64,
        )
        yaw_rad = as_float(row, "debug_world_yaw_rad")
        if not np.all(np.isfinite(position)) or not math.isfinite(yaw_rad):
            continue

        t = as_float(row, "timestamp_s", 0.0)
        velocity_world = finite_velocity_world(row, yaw_rad)
        if last_velocity is not None and last_t is not None and t > last_t:
            acceleration_world = (velocity_world - last_velocity) / max(t - last_t, 1e-3)
        else:
            acceleration_world = np.zeros(3, dtype=np.float64)
        last_velocity = velocity_world
        last_t = t

        gate = choose_gate(row, gates)
        past_gate_m = float((position - gate.center) @ gate.normal_enu)
        if past_gate_m > max_past_gate_m:
            continue
        reference_idx = nearest_reference_index(
            reference_samples,
            position=position,
            gate_index=gate.index,
        )
        reference_sample = reference_samples[reference_idx]
        off_reference_m = float(np.linalg.norm(position - reference_sample.position))
        if off_reference_m > max_reference_error_m:
            continue
        rejoin_lookahead_m = float(
            np.clip(2.4 + 0.35 * off_reference_m, 2.2, lookahead_m)
        )
        rejoin_sample = lookahead_reference(
            reference_samples,
            start_idx=reference_idx,
            lookahead_m=rejoin_lookahead_m,
        )
        command, target, distance = relabel_command(
            position=position,
            yaw_rad=yaw_rad,
            velocity_world=velocity_world,
            gate=gate,
            lookahead_m=lookahead_m,
            observed_bearing_h=as_float(row, "bearing_h_rad", math.nan),
            teacher=teacher,
            reference_position=reference_sample.position,
            reference_velocity_world=reference_sample.velocity_world,
            rejoin_target=rejoin_sample.position,
        )

        out_row = {name: "" for name in FIELDNAMES}
        out_row.update(
            {
                "timestamp_s": f"{t:.6f}",
                "frame_fresh": row.get("frame_fresh", "1"),
                "course": course_name,
                "episode_id": episode_id,
                "teacher_phase": f"closed_loop_{teacher}",
                "command_source": command_source or "closed_loop",
                "last_gate_passed": str(as_int(row, "last_gate_passed", -1)),
                "next_gate_index": str(gate.index),
                "mode": mode,
                "confidence": f"{as_float(row, 'confidence', 0.0):.6f}",
                "bearing_h_rad": f"{as_float(row, 'bearing_h_rad', 0.0):.6f}",
                "bearing_v_rad": f"{as_float(row, 'bearing_v_rad', 0.0):.6f}",
                "distance_m": row.get("distance_m", ""),
                "pixel_x": row.get("pixel_x", ""),
                "pixel_y": row.get("pixel_y", ""),
                "apparent_size_px": row.get("apparent_size_px", ""),
                "gate_age_s": row.get("gate_age_s", "0.000000"),
                "body_forward_elevation_rad": row.get("body_forward_elevation_rad", "0.000000"),
                "body_vx_m_s": f"{as_float(row, 'body_vx_m_s', 0.0):.6f}",
                "body_vy_m_s": f"{as_float(row, 'body_vy_m_s', 0.0):.6f}",
                "body_vz_m_s": f"{as_float(row, 'body_vz_m_s', 0.0):.6f}",
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
                "teacher_target_x_m": f"{target[0]:.6f}",
                "teacher_target_y_m": f"{target[1]:.6f}",
                "teacher_target_z_m": f"{target[2]:.6f}",
                "teacher_next_gate_index": str(gate.index),
                "teacher_next_gate_x_m": f"{gate.center[0]:.6f}",
                "teacher_next_gate_y_m": f"{gate.center[1]:.6f}",
                "teacher_next_gate_z_m": f"{gate.center[2]:.6f}",
                "teacher_next_gate_yaw_deg": f"{gate.yaw_deg:.6f}",
                "teacher_heading_rate_rad_s": f"{command[2]:.6f}",
                "teacher_world_vx_m_s": f"{velocity_world[0]:.6f}",
                "teacher_world_vy_m_s": f"{velocity_world[1]:.6f}",
                "teacher_world_vz_m_s": f"{velocity_world[2]:.6f}",
                "teacher_world_ax_m_s2": f"{acceleration_world[0]:.6f}",
                "teacher_world_ay_m_s2": f"{acceleration_world[1]:.6f}",
                "teacher_world_az_m_s2": f"{acceleration_world[2]:.6f}",
                "world_x_m": f"{position[0]:.6f}",
                "world_y_m": f"{position[1]:.6f}",
                "world_z_m": f"{position[2]:.6f}",
                "world_yaw_rad": f"{yaw_rad:.6f}",
            }
        )
        out.append(out_row)
        prev_command = command
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--course", choices=course_names(), required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--lookahead-m", type=float, default=3.5)
    parser.add_argument(
        "--teacher",
        choices=("baseline", "rejoin"),
        default="rejoin",
        help="Offline relabeling teacher. 'rejoin' uses a local reference-line rejoin target.",
    )
    parser.add_argument("--reference-speed-m-s", type=float, default=6.0)
    parser.add_argument("--reference-samples-per-segment", type=int, default=120)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument(
        "--max-past-gate-m",
        type=float,
        default=2.0,
        help="Drop rows this far past an unpassed gate; they are post-miss recovery, not pass-through labels.",
    )
    parser.add_argument(
        "--max-reference-error-m",
        type=float,
        default=3.0,
        help="Drop closed-loop states farther than this from the reference line.",
    )
    parser.add_argument(
        "--allowed-state-command-sources",
        default="",
        help=(
            "Comma-separated command_source values to keep from the rollout "
            "(for example learned). Empty keeps all sources."
        ),
    )
    args = parser.parse_args()

    rows = read_rows(args.trace)
    episode_id = args.episode_id or f"{args.course}:closed_loop:{args.trace.parent.name}"
    allowed_sources = split_csv(args.allowed_state_command_sources)
    out_rows = relabel_rows(
        rows,
        course_name=args.course,
        episode_id=episode_id,
        lookahead_m=args.lookahead_m,
        min_confidence=args.min_confidence,
        max_past_gate_m=args.max_past_gate_m,
        max_reference_error_m=args.max_reference_error_m,
        allowed_state_command_sources=allowed_sources or None,
        teacher=args.teacher,
        reference_speed_m_s=args.reference_speed_m_s,
        reference_samples_per_segment=args.reference_samples_per_segment,
    )
    if not out_rows:
        raise SystemExit("no rows relabeled; check trace modes/confidence/debug-world columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"trace={args.trace}")
    print(f"dataset={args.out}")
    print(f"course={args.course}")
    print(f"episode_id={episode_id}")
    print(f"teacher={args.teacher}")
    print(
        "allowed_state_command_sources="
        f"{','.join(sorted(allowed_sources)) if allowed_sources else 'all'}"
    )
    print(f"max_reference_error_m={args.max_reference_error_m:g}")
    print(f"rows_in={len(rows)}")
    print(f"rows_relabelled={len(out_rows)}")
    print("student_inputs=original FPV/tracker/telemetry trace columns")
    print("privileged_teacher=debug_world_* plus known debug course geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

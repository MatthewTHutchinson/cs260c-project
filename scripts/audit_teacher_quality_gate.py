#!/usr/bin/env python3
"""Fail-fast audit for privileged teacher datasets before BC/T4 training.

This is stricter than ``audit_privileged_teacher_dataset.py``. The summary audit
is useful for plots and ranges; this quality gate answers whether a generated
teacher CSV is safe enough to spend training time on.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
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

from learning.datasets import FeatureSpec, SEQUENCE_FEATURE_COLUMNS, TARGET_COLUMNS


PRIVILEGED_PREFIXES = ("debug_world_", "world_", "teacher_", "reference_")
PRIVILEGED_EXACT = {
    "gate_id",
    "gate_index",
    "gate_center_x_m",
    "gate_center_y_m",
    "gate_center_z_m",
}
DERIVED_FEATURE_PREFIXES = ("mode_",)
DERIVED_FEATURE_SUFFIXES = ("_delta",)
TEACHER_COMMAND_COLUMNS = (
    "teacher_roll_rate_rad_s",
    "teacher_pitch_rate_rad_s",
    "teacher_yaw_rate_rad_s",
    "teacher_thrust_norm",
)
COMMAND_LIMITS = {
    "teacher_roll_rate_rad_s": 0.70,
    "teacher_pitch_rate_rad_s": 0.80,
    "teacher_yaw_rate_rad_s": 1.20,
}
REQUIRED_COLUMNS = (
    "course",
    "episode_id",
    "teacher_phase",
    "command_source",
    "mode",
    "confidence",
    "bearing_h_rad",
    "bearing_v_rad",
    "distance_m",
    "body_vx_m_s",
    "body_vy_m_s",
    "body_vz_m_s",
    "teacher_target_x_m",
    "teacher_target_y_m",
    "teacher_target_z_m",
    "teacher_next_gate_index",
    "teacher_next_gate_x_m",
    "teacher_next_gate_y_m",
    "teacher_next_gate_z_m",
    "world_x_m",
    "world_y_m",
    "world_z_m",
    "world_yaw_rad",
    *TEACHER_COMMAND_COLUMNS,
)


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


def is_privileged(name: str) -> bool:
    return name in PRIVILEGED_EXACT or any(name.startswith(prefix) for prefix in PRIVILEGED_PREFIXES)


def is_derived_feature(name: str) -> bool:
    return (
        name == "has_distance"
        or any(name.startswith(prefix) for prefix in DERIVED_FEATURE_PREFIXES)
        or any(name.endswith(suffix) for suffix in DERIVED_FEATURE_SUFFIXES)
    )


def load_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return fieldnames, rows


def selected_feature_failures(fieldnames: tuple[str, ...]) -> list[str]:
    spec = FeatureSpec.default(include_prev_command=False, include_sequence_features=False)
    features = spec.feature_names
    field_set = set(fieldnames)
    failures: list[str] = []
    privileged_features = sorted(name for name in features if is_privileged(name))
    if privileged_features:
        failures.append(f"selected_privileged_features={','.join(privileged_features)}")
    sequence_features = sorted(name for name in SEQUENCE_FEATURE_COLUMNS if name in features)
    if sequence_features:
        failures.append(f"selected_sequence_features={','.join(sequence_features)}")
    prev_command_features = sorted(f"prev_{name}" for name in TARGET_COLUMNS if f"prev_{name}" in features)
    if prev_command_features:
        failures.append(f"selected_prev_command_features={','.join(prev_command_features)}")
    missing = sorted(name for name in features if name not in field_set and not is_derived_feature(name))
    if missing:
        failures.append(f"selected_trace_columns_missing={','.join(missing)}")
    return failures


def target_body_components(row: dict[str, str]) -> tuple[float, float, float]:
    yaw = as_float(row, "world_yaw_rad")
    dx = as_float(row, "teacher_target_x_m") - as_float(row, "world_x_m")
    dy = as_float(row, "teacher_target_y_m") - as_float(row, "world_y_m")
    dz = as_float(row, "teacher_target_z_m") - as_float(row, "world_z_m")
    c = math.cos(yaw)
    s = math.sin(yaw)
    body_x = c * dx + s * dy
    body_y = -s * dx + c * dy
    return body_x, body_y, dz


def gate_center_errors(rows: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    nominal_rows = [row for row in rows if row.get("teacher_phase") == "nominal"]
    positions_by_course: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    gate_centers: dict[tuple[str, str], tuple[float, float, float]] = {}
    for row in nominal_rows:
        course = row.get("course", "unknown")
        positions_by_course[course].append(
            (
                as_float(row, "world_x_m"),
                as_float(row, "world_y_m"),
                as_float(row, "world_z_m"),
            )
        )
        gate_centers[(course, row.get("teacher_next_gate_index", ""))] = (
            as_float(row, "teacher_next_gate_x_m"),
            as_float(row, "teacher_next_gate_y_m"),
            as_float(row, "teacher_next_gate_z_m"),
        )

    errors: list[tuple[str, str, float]] = []
    for (course, gate_idx), center in sorted(gate_centers.items()):
        positions = np.asarray(positions_by_course[course], dtype=np.float64)
        center_np = np.asarray(center, dtype=np.float64)
        distances = np.linalg.norm(positions - center_np, axis=1)
        errors.append((course, gate_idx, float(np.nanmin(distances))))
    return errors


def saturation_percent(rows: list[dict[str, str]], column: str, limit: float) -> float:
    values = np.asarray([as_float(row, column) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 100.0
    return float(np.mean(np.abs(values) >= limit - 1e-6) * 100.0)


def thrust_saturation_percent(rows: list[dict[str, str]]) -> float:
    values = np.asarray([as_float(row, "teacher_thrust_norm") for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 100.0
    return float(np.mean((values <= 0.35 + 1e-6) | (values >= 0.90 - 1e-6)) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--require-courses", default="easy,circular_arc,s_curve")
    parser.add_argument("--require-phases", default="launch,nominal,off_nominal")
    parser.add_argument("--allowed-command-sources", default="teacher")
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--max-gate-center-error-m", type=float, default=0.25)
    parser.add_argument("--max-backward-lookahead-pct", type=float, default=1.0)
    parser.add_argument("--min-lookahead-forward-p1-m", type=float, default=-0.05)
    parser.add_argument("--min-off-nominal-align-pct", type=float, default=85.0)
    parser.add_argument("--max-command-saturation-pct", type=float, default=45.0)
    parser.add_argument("--alignment-lateral-threshold-m", type=float, default=0.15)
    args = parser.parse_args()

    fieldnames, rows = load_rows(args.trace)
    field_set = set(fieldnames)
    failures: list[str] = []
    warnings: list[str] = []

    missing_required = sorted(name for name in REQUIRED_COLUMNS if name not in field_set)
    if missing_required:
        failures.append(f"missing_required_columns={','.join(missing_required)}")
    failures.extend(selected_feature_failures(fieldnames))

    courses = Counter(row.get("course", "unknown") or "unknown" for row in rows)
    phases = Counter(row.get("teacher_phase", "unknown") or "unknown" for row in rows)
    command_sources = Counter(row.get("command_source", "unknown") or "unknown" for row in rows)
    required_courses = set(split_csv(args.require_courses))
    required_phases = set(split_csv(args.require_phases))
    allowed_sources = set(split_csv(args.allowed_command_sources))

    if len(rows) < args.min_rows:
        failures.append(f"too_few_rows={len(rows)} min={args.min_rows}")
    missing_courses = sorted(course for course in required_courses if course not in courses)
    if missing_courses:
        failures.append(f"missing_required_courses={','.join(missing_courses)}")
    missing_phases = sorted(phase for phase in required_phases if phase not in phases)
    if missing_phases:
        failures.append(f"missing_required_phases={','.join(missing_phases)}")
    disallowed_sources = sorted(source for source in command_sources if source not in allowed_sources)
    if disallowed_sources:
        failures.append(f"disallowed_command_sources={','.join(disallowed_sources)}")

    target_forward = np.asarray([target_body_components(row)[0] for row in rows], dtype=np.float64)
    finite_forward = target_forward[np.isfinite(target_forward)]
    if finite_forward.size == 0:
        failures.append("lookahead_forward_missing")
    else:
        backward_pct = float(np.mean(finite_forward < -1e-6) * 100.0)
        p1_forward = float(np.nanpercentile(finite_forward, 1))
        if backward_pct > args.max_backward_lookahead_pct:
            failures.append(
                f"backward_lookahead_pct={backward_pct:.2f} max={args.max_backward_lookahead_pct:.2f}"
            )
        if p1_forward < args.min_lookahead_forward_p1_m:
            failures.append(
                f"lookahead_forward_p1_m={p1_forward:.3f} min={args.min_lookahead_forward_p1_m:.3f}"
            )

    gate_errors = gate_center_errors(rows)
    max_gate_error = max((error for _course, _gate, error in gate_errors), default=float("inf"))
    if max_gate_error > args.max_gate_center_error_m:
        worst = sorted(gate_errors, key=lambda item: item[2], reverse=True)[:5]
        failures.append(
            "gate_center_error_m="
            + ";".join(f"{course}:{gate}:{error:.3f}" for course, gate, error in worst)
        )

    off_nominal_rows = [row for row in rows if row.get("teacher_phase") == "off_nominal"]
    align_checks = []
    for row in off_nominal_rows:
        _body_x, body_y, _body_z = target_body_components(row)
        if abs(body_y) < args.alignment_lateral_threshold_m:
            continue
        yaw_rate = as_float(row, "teacher_yaw_rate_rad_s")
        align_checks.append(yaw_rate * body_y >= -0.03)
    if required_phases and "off_nominal" in required_phases and not align_checks:
        failures.append("off_nominal_alignment_rows=0")
    elif align_checks:
        align_pct = float(np.mean(align_checks) * 100.0)
        if align_pct < args.min_off_nominal_align_pct:
            failures.append(
                f"off_nominal_yaw_target_alignment_pct={align_pct:.1f} "
                f"min={args.min_off_nominal_align_pct:.1f}"
            )

    course_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        course_rows[row.get("course", "unknown") or "unknown"].append(row)
    saturation_failures: list[str] = []
    for course, course_group in sorted(course_rows.items()):
        for column, limit in COMMAND_LIMITS.items():
            sat_pct = saturation_percent(course_group, column, limit)
            if sat_pct > args.max_command_saturation_pct:
                saturation_failures.append(f"{course}:{column}:{sat_pct:.1f}")
        thrust_sat = thrust_saturation_percent(course_group)
        if thrust_sat > args.max_command_saturation_pct:
            saturation_failures.append(f"{course}:teacher_thrust_norm:{thrust_sat:.1f}")
    if saturation_failures:
        failures.append(f"command_saturation_pct={','.join(saturation_failures[:8])}")
        if len(saturation_failures) > 8:
            warnings.append(f"additional_saturation_failures={len(saturation_failures) - 8}")

    print(f"trace={args.trace}")
    print(f"rows={len(rows)}")
    print(f"courses={len(courses)} required={','.join(sorted(required_courses)) if required_courses else 'none'}")
    print(f"phases={','.join(f'{key}:{value}' for key, value in sorted(phases.items()))}")
    print(
        "command_sources="
        f"{','.join(f'{key}:{value}' for key, value in sorted(command_sources.items()))}"
    )
    if finite_forward.size:
        print(
            f"lookahead_forward_p1_m={float(np.nanpercentile(finite_forward, 1)):.3f} "
            f"backward_pct={float(np.mean(finite_forward < -1e-6) * 100.0):.2f}"
        )
    print(f"max_gate_center_error_m={max_gate_error:.3f}")
    if align_checks:
        print(f"off_nominal_yaw_target_alignment_pct={float(np.mean(align_checks) * 100.0):.1f}")
    print(
        "selected_student_features=no_prev_command,no_sequence,no_privileged "
        f"feature_count={len(FeatureSpec.default(include_prev_command=False, include_sequence_features=False).feature_names)}"
    )
    for warning in warnings:
        print(f"warning={warning}")
    if failures:
        print("verdict=FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("verdict=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

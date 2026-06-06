#!/usr/bin/env python3
"""Audit horizontal commands and post-gate reacquisition control behavior."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
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

from algorithm.control_adapter import to_betaflight_rc_fields
from algorithm.frames import body_forward_elevation_from_quat_xyzw
from algorithm.reactive_controller import ReactiveGateController
from algorithm.types import GateEstimate, TrackMode, VehicleTelemetry


def gate(
    *,
    bearing_h: float = 0.0,
    bearing_v: float = 0.0,
    distance: float = 4.5,
    confidence: float = 0.85,
    mode: TrackMode = TrackMode.DETECTED,
) -> GateEstimate:
    return GateEstimate(
        bearing_h_rad=bearing_h,
        bearing_v_rad=bearing_v,
        distance_m=distance,
        confidence=confidence,
        pixel_center=(320.0 + 320.0 * np.tan(bearing_h), 180.0 - 320.0 * np.tan(bearing_v)),
        mode=mode,
    )


def quat_for_forward_elevation(elevation_rad: float) -> np.ndarray:
    """Build an `[x, y, z, w]` quat whose body +X axis has this elevation."""
    half = elevation_rad / 2.0
    return np.array([0.0, -np.sin(half), 0.0, np.cos(half)], dtype=np.float64)


def command_row(
    label: str,
    controller: ReactiveGateController,
    estimate: GateEstimate,
    telemetry: VehicleTelemetry | None = None,
) -> dict[str, float | str | int]:
    telemetry = telemetry or VehicleTelemetry()
    command = controller.compute(estimate, telemetry)
    rc = to_betaflight_rc_fields(command)
    body_forward_elevation = 0.0
    if telemetry.attitude_quat is not None:
        body_forward_elevation = body_forward_elevation_from_quat_xyzw(telemetry.attitude_quat)
    return {
        "label": label,
        "bearing_h": estimate.bearing_h_rad,
        "bearing_v": estimate.bearing_v_rad,
        "body_forward_elevation": body_forward_elevation,
        "distance": estimate.distance_m or float("nan"),
        "roll": command.roll_rate_rad_s,
        "pitch": command.pitch_rate_rad_s,
        "yaw": command.yaw_rate_rad_s,
        "thrust": command.thrust_norm,
        "rc_roll": rc["roll"],
        "rc_pitch": rc["pitch"],
        "rc_yaw": rc["yaw"],
    }


def print_command_row(row: dict[str, float | str | int]) -> None:
    print(
        "synthetic="
        f"{row['label']} "
        f"bearing=({float(row['bearing_h']):+.3f},{float(row['bearing_v']):+.3f}) "
        f"body_forward_elev={float(row['body_forward_elevation']):+.3f} "
        f"distance={float(row['distance']):.2f} "
        f"roll={float(row['roll']):+.3f} "
        f"pitch={float(row['pitch']):+.3f} "
        f"yaw={float(row['yaw']):+.3f} "
        f"thrust={float(row['thrust']):.3f} "
        f"rc=({int(row['rc_roll'])},{int(row['rc_pitch'])},{int(row['rc_yaw'])})"
    )


def load_trace(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = -999) -> int:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def estimate_from_trace(row: dict[str, str]) -> GateEstimate | None:
    mode_raw = row.get("mode", "")
    if mode_raw not in TrackMode._value2member_map_:
        return None
    mode = TrackMode(mode_raw)
    if mode not in {TrackMode.DETECTED, TrackMode.TRACKED, TrackMode.COMMIT}:
        return None

    distance = as_float(row, "distance_m")
    pixel_x = as_float(row, "pixel_x")
    pixel_y = as_float(row, "pixel_y")
    return GateEstimate(
        bearing_h_rad=as_float(row, "bearing_h_rad", 0.0),
        bearing_v_rad=as_float(row, "bearing_v_rad", 0.0),
        distance_m=None if np.isnan(distance) else distance,
        confidence=as_float(row, "confidence", 0.0),
        pixel_center=None if np.isnan(pixel_x) or np.isnan(pixel_y) else (pixel_x, pixel_y),
        mode=mode,
    )


def trace_after_first_pass(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    passed = [row for row in rows if as_int(row, "last_gate_passed") >= 0]
    if not passed:
        return []
    start_t = as_float(passed[0], "timestamp_s", 0.0)
    return [
        row
        for row in passed
        if as_float(row, "timestamp_s", 0.0) <= start_t + 3.0
    ]


def summarize_trace(controller: ReactiveGateController, rows: list[dict[str, str]]) -> list[str]:
    post = trace_after_first_pass(rows)
    if not rows:
        return ["trace_status=missing_or_empty"]
    if not post:
        return ["trace_status=no_gate0_pass_in_trace"]

    estimates = [(row, estimate_from_trace(row)) for row in post]
    estimates = [(row, estimate) for row, estimate in estimates if estimate is not None]
    if not estimates:
        return ["trace_status=no_usable_post_gate_estimates"]

    observed_pitch = np.array([as_float(row, "pitch_rate_rad_s") for row, _ in estimates])
    observed_yaw = np.array([as_float(row, "yaw_rate_rad_s") for row, _ in estimates])
    observed_roll = np.array([as_float(row, "roll_rate_rad_s") for row, _ in estimates])
    bearings_h = np.array([estimate.bearing_h_rad for _, estimate in estimates])
    body_forward = np.array(
        [as_float(row, "body_forward_elevation_rad") for row, _ in estimates]
    )
    distances = np.array(
        [
            np.nan if estimate.distance_m is None else estimate.distance_m
            for _, estimate in estimates
        ]
    )
    recomputed = np.array(
        [
            controller.compute(
                estimate,
                VehicleTelemetry(
                    rpy_rad=(
                        None
                        if np.isnan(as_float(row, "body_forward_elevation_rad"))
                        else np.array([0.0, as_float(row, "body_forward_elevation_rad"), 0.0])
                    )
                ),
            ).pitch_rate_rad_s
            for row, estimate in estimates
        ]
    )

    active = np.abs(bearings_h) > 0.02
    yaw_aligned = float(np.mean(np.sign(bearings_h[active]) == np.sign(observed_yaw[active]))) if np.any(active) else np.nan
    roll_aligned = float(np.mean(np.sign(bearings_h[active]) == np.sign(observed_roll[active]))) if np.any(active) else np.nan

    modes = Counter(row.get("mode", "") for row, _ in estimates)
    return [
        "trace_status=post_gate_reacquisition_window",
        f"post_gate_rows={len(post)}",
        f"post_gate_modes={dict(modes)}",
        f"post_gate_bearing_h_median={float(np.nanmedian(bearings_h)):.6f}",
        f"post_gate_body_forward_elevation_median={float(np.nanmedian(body_forward)):.6f}",
        f"post_gate_distance_median={float(np.nanmedian(distances)):.6f}",
        f"post_gate_observed_pitch_median={float(np.nanmedian(observed_pitch)):.6f}",
        f"post_gate_recomputed_pitch_median={float(np.nanmedian(recomputed)):.6f}",
        f"post_gate_yaw_alignment={yaw_aligned:.3f}",
        f"post_gate_roll_alignment={roll_aligned:.3f}",
        f"latest_last_gate_passed={as_int(post[-1], 'last_gate_passed')}",
        f"latest_next_gate_index={as_int(post[-1], 'next_gate_index')}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/elodin_pilot_trace_editor.csv"))
    args = parser.parse_args()

    controller = ReactiveGateController()
    tilt = controller.gains.camera_tilt_up_rad
    synthetic = [
        command_row("left_gate", controller, gate(bearing_h=-0.25, bearing_v=-tilt)),
        command_row("center_body_aligned", controller, gate(bearing_h=0.0, bearing_v=-tilt)),
        command_row("right_gate", controller, gate(bearing_h=0.25, bearing_v=-tilt)),
        command_row("optical_center_no_attitude", controller, gate(bearing_h=0.25, bearing_v=0.0, distance=2.4, confidence=0.76)),
        command_row(
            "optical_center_pitched_down",
            controller,
            gate(bearing_h=0.25, bearing_v=0.0, distance=2.4, confidence=0.76),
            VehicleTelemetry(attitude_quat=quat_for_forward_elevation(-0.31)),
        ),
        command_row("top_edge_gate", controller, gate(bearing_h=0.0, bearing_v=0.44)),
    ]

    for row in synthetic:
        print_command_row(row)

    trace_rows = load_trace(args.trace)
    for line in summarize_trace(controller, trace_rows):
        print(line)

    failures: list[str] = []
    by_label = {str(row["label"]): row for row in synthetic}
    left = by_label["left_gate"]
    center = by_label["center_body_aligned"]
    right = by_label["right_gate"]
    no_attitude = by_label["optical_center_no_attitude"]
    reacquire = by_label["optical_center_pitched_down"]
    top = by_label["top_edge_gate"]
    if float(right["yaw"]) <= 0.0 or int(right["rc_yaw"]) <= 1500:
        failures.append("right gate should command positive yaw / RC yaw above center")
    if float(right["roll"]) <= 0.0 or int(right["rc_roll"]) <= 1500:
        failures.append("right gate should command positive roll / RC roll above center")
    if float(left["yaw"]) >= 0.0 or int(left["rc_yaw"]) >= 1500:
        failures.append("left gate should command negative yaw / RC yaw below center")
    if float(left["roll"]) >= 0.0 or int(left["rc_roll"]) >= 1500:
        failures.append("left gate should command negative roll / RC roll below center")
    if abs(float(center["yaw"])) > 1e-9 or abs(float(center["roll"])) > 1e-9:
        failures.append("centered gate should not command lateral/yaw correction")
    if float(center["pitch"]) >= -0.01:
        failures.append("center body-aligned gate should command forward pitch")
    if float(no_attitude["pitch"]) < -0.01:
        failures.append("optical-center gate without attitude should stay conservative")
    if float(reacquire["pitch"]) >= -0.01:
        failures.append("pitched-down optical-center reacquisition gate should retain forward pitch")
    if abs(float(top["pitch"])) >= abs(float(center["pitch"])):
        failures.append("top-edge gate should suppress forward pitch relative to centered gate")

    if failures:
        print("verdict=FAIL")
        for failure in failures:
            print(f"failure={failure}")
        return 1

    print("verdict=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit camera/control sign conventions for the Elodin practice harness."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


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

from algorithm.control_adapter import to_betaflight_rc_fields
from algorithm.gate_detector import CameraParams
from algorithm.reactive_controller import ReactiveGateController
from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry


def expected_pixel_y(
    *,
    gate_x_m: float,
    gate_z_m: float,
    drone_x_m: float,
    drone_z_m: float,
    camera_tilt_up_deg: float,
    camera_params: CameraParams,
) -> tuple[float, float, float]:
    """Return body elevation and expected y if camera is up/down by tilt."""
    body_elev_rad = math.atan2(gate_z_m - drone_z_m, gate_x_m - drone_x_m)
    tilt_rad = math.radians(camera_tilt_up_deg)
    bearing_if_up = body_elev_rad - tilt_rad
    bearing_if_down = body_elev_rad + tilt_rad
    y_if_up = camera_params.cy - camera_params.fy * math.tan(bearing_if_up)
    y_if_down = camera_params.cy - camera_params.fy * math.tan(bearing_if_down)
    return math.degrees(body_elev_rad), y_if_up, y_if_down


def first_non_search(trace_path: Path) -> dict[str, str] | None:
    if not trace_path.exists():
        return None
    with trace_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") and row.get("mode") != "search":
                return row
    return None


def elodin_pitch_offset(camera_path: Path) -> str:
    if not camera_path.exists():
        return "missing"
    text = camera_path.read_text()
    match = re.search(r"ELODIN_ROT_OFFSET_PITCH_DEG\s*=\s*([^\n]+)", text)
    if not match:
        return "not_found"
    return match.group(1).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/elodin_pilot_trace_editor.csv"))
    parser.add_argument(
        "--elodin-camera",
        type=Path,
        default=Path("/Users/matthewhutchinson/dev/elodin-ai-grand-prix/sim/camera.py"),
    )
    parser.add_argument("--gate-x-m", type=float, default=10.0)
    parser.add_argument("--gate-z-m", type=float, default=1.8)
    parser.add_argument("--drone-x-m", type=float, default=0.0)
    parser.add_argument("--drone-z-m", type=float, default=0.34)
    parser.add_argument("--camera-tilt-up-deg", type=float, default=20.0)
    args = parser.parse_args()

    cam = CameraParams()
    body_elev_deg, y_if_up, y_if_down = expected_pixel_y(
        gate_x_m=args.gate_x_m,
        gate_z_m=args.gate_z_m,
        drone_x_m=args.drone_x_m,
        drone_z_m=args.drone_z_m,
        camera_tilt_up_deg=args.camera_tilt_up_deg,
        camera_params=cam,
    )

    forward_rc = to_betaflight_rc_fields(
        RacingCommand(pitch_rate_rad_s=-0.25, thrust_norm=0.55)
    )
    backward_rc = to_betaflight_rc_fields(
        RacingCommand(pitch_rate_rad_s=0.25, thrust_norm=0.55)
    )

    controller = ReactiveGateController()
    tilt_rad = controller.gains.camera_tilt_up_rad

    high_gate = GateEstimate(
        bearing_h_rad=0.0,
        bearing_v_rad=0.44,
        distance_m=4.75,
        confidence=0.85,
        pixel_center=(320.0, 30.0),
        apparent_size_px=101.0,
        mode=TrackMode.DETECTED,
    )
    body_aligned_gate = GateEstimate(
        bearing_h_rad=0.0,
        bearing_v_rad=-tilt_rad,
        distance_m=4.75,
        confidence=0.85,
        pixel_center=(320.0, 296.0),
        apparent_size_px=101.0,
        mode=TrackMode.DETECTED,
    )
    below_optical_gate = GateEstimate(
        bearing_h_rad=0.0,
        bearing_v_rad=-0.20,
        distance_m=4.95,
        confidence=0.85,
        pixel_center=(320.0, 244.0),
        apparent_size_px=97.0,
        mode=TrackMode.DETECTED,
    )
    right_gate = GateEstimate(
        bearing_h_rad=0.20,
        bearing_v_rad=-tilt_rad,
        distance_m=4.95,
        confidence=0.85,
        pixel_center=(384.0, 296.0),
        apparent_size_px=97.0,
        mode=TrackMode.DETECTED,
    )
    high_cmd = controller.compute(high_gate, VehicleTelemetry())
    body_aligned_cmd = controller.compute(body_aligned_gate, VehicleTelemetry())
    below_optical_cmd = controller.compute(below_optical_gate, VehicleTelemetry())
    right_cmd = controller.compute(right_gate, VehicleTelemetry())
    right_rc = to_betaflight_rc_fields(right_cmd)
    search_gate = GateEstimate(mode=TrackMode.SEARCH)
    search_settle_cmd = controller.compute(
        search_gate,
        VehicleTelemetry(timestamp_s=1.0),
    )
    search_scan_cmd = controller.compute(
        search_gate,
        VehicleTelemetry(timestamp_s=1.0 + controller.gains.search_settle_s + 0.1),
    )
    pitched_search_cmd = controller.compute(
        search_gate,
        VehicleTelemetry(
            rpy_rad=np.array([0.0, -0.25, 0.0]),
            timestamp_s=1.0 + controller.gains.search_settle_s + 0.2,
        ),
    )
    moving_search_cmd = controller.compute(
        search_gate,
        VehicleTelemetry(
            rpy_rad=np.array([0.0, 0.0, 0.0]),
            linear_velocity_m_s=np.array([2.5, 1.0, 0.0]),
            timestamp_s=1.0 + controller.gains.search_settle_s + 0.3,
        ),
    )

    observed = first_non_search(args.trace)

    print(f"spec_camera_tilt_up_deg={args.camera_tilt_up_deg}")
    print(f"controller_camera_tilt_up_deg={math.degrees(controller.gains.camera_tilt_up_rad):.1f}")
    print(f"elodin_rot_offset_pitch_expr={elodin_pitch_offset(args.elodin_camera)}")
    print(f"assumed_geometry_gate=({args.gate_x_m:.2f}, z={args.gate_z_m:.2f})")
    print(f"assumed_geometry_drone=({args.drone_x_m:.2f}, z={args.drone_z_m:.2f})")
    print(f"body_elevation_to_gate_deg={body_elev_deg:.2f}")
    print(f"expected_pixel_y_if_camera_up={y_if_up:.1f}")
    print(f"expected_pixel_y_if_camera_down={y_if_down:.1f}")
    if observed is None:
        print(f"observed_trace={args.trace}: no non-search rows")
    else:
        y = observed.get("pixel_y", "")
        print(
            "observed_first_non_search="
            f"tick={observed.get('tick', '')} mode={observed.get('mode', '')} "
            f"pixel_y={y} bearing_v={observed.get('bearing_v_rad', '')}"
        )
        try:
            observed_y = float(y)
            closer = "up" if abs(observed_y - y_if_up) < abs(observed_y - y_if_down) else "down"
            print(f"observed_pixel_y_closer_to_camera={closer}")
        except ValueError:
            pass

    print("detector_bearing_v_sign=positive means pixel_y is above image center")
    print(f"forward_internal_pitch=-0.25 rc_pitch={forward_rc['pitch']}")
    print(f"backward_internal_pitch=+0.25 rc_pitch={backward_rc['pitch']}")
    print(
        "high_gate_command="
        f"pitch={high_cmd.pitch_rate_rad_s:.3f} thrust={high_cmd.thrust_norm:.3f}"
    )
    print(
        "below_optical_center_command="
        f"body_elevation={below_optical_gate.bearing_v_rad + controller.gains.camera_tilt_up_rad:.3f} "
        f"pitch={below_optical_cmd.pitch_rate_rad_s:.3f} "
        f"thrust={below_optical_cmd.thrust_norm:.3f}"
    )
    print(
        "body_aligned_gate_command="
        f"pitch={body_aligned_cmd.pitch_rate_rad_s:.3f} "
        f"thrust={body_aligned_cmd.thrust_norm:.3f}"
    )
    print(
        "right_gate_command="
        f"roll={right_cmd.roll_rate_rad_s:.3f} yaw={right_cmd.yaw_rate_rad_s:.3f} "
        f"rc_roll={right_rc['roll']} rc_yaw={right_rc['yaw']}"
    )
    print(
        "search_command="
        f"settle_yaw={search_settle_cmd.yaw_rate_rad_s:.3f} "
        f"scan_yaw={search_scan_cmd.yaw_rate_rad_s:.3f} "
        f"roll={search_scan_cmd.roll_rate_rad_s:.3f} "
        f"pitch={search_scan_cmd.pitch_rate_rad_s:.3f} "
        f"thrust={search_scan_cmd.thrust_norm:.3f}"
    )
    print(
        "pitched_search_command="
        f"pitch={pitched_search_cmd.pitch_rate_rad_s:.3f} "
        f"yaw={pitched_search_cmd.yaw_rate_rad_s:.3f} "
        f"thrust={pitched_search_cmd.thrust_norm:.3f}"
    )
    print(
        "moving_search_command="
        f"roll={moving_search_cmd.roll_rate_rad_s:.3f} "
        f"pitch={moving_search_cmd.pitch_rate_rad_s:.3f} "
        f"yaw={moving_search_cmd.yaw_rate_rad_s:.3f} "
        f"thrust={moving_search_cmd.thrust_norm:.3f}"
    )

    failures = []
    if "CAM_TILT_UP_DEG" not in elodin_pitch_offset(args.elodin_camera):
        failures.append("elodin camera pitch offset is not tied to CAM_TILT_UP_DEG")
    if "-" not in elodin_pitch_offset(args.elodin_camera):
        failures.append("elodin camera pitch offset should be negative for upward tilt")
    if forward_rc["pitch"] <= 1500:
        failures.append("forward internal pitch should map to RC pitch above center")
    if backward_rc["pitch"] >= 1500:
        failures.append("backward internal pitch should map to RC pitch below center")
    if abs(math.degrees(controller.gains.camera_tilt_up_rad) - args.camera_tilt_up_deg) > 1e-6:
        failures.append("controller camera tilt should match the spec upward tilt")
    if below_optical_cmd.thrust_norm <= controller.gains.hover_thrust:
        failures.append("below-optical first gate should still climb after camera-tilt compensation")
    if high_cmd.thrust_norm <= body_aligned_cmd.thrust_norm:
        failures.append("high gate should command more thrust than body-aligned gate")
    if abs(high_cmd.pitch_rate_rad_s) >= abs(body_aligned_cmd.pitch_rate_rad_s):
        failures.append("high gate should suppress forward pitch relative to body-aligned gate")
    if right_cmd.yaw_rate_rad_s <= 0.0 or right_rc["yaw"] <= 1500:
        failures.append("right-of-center gate should command positive internal yaw and RC yaw above center")
    if right_cmd.roll_rate_rad_s <= 0.0 or right_rc["roll"] <= 1500:
        failures.append("right-of-center gate should command positive internal roll and RC roll above center")
    if abs(search_settle_cmd.yaw_rate_rad_s) > 1e-9:
        failures.append("search should settle at hover before starting yaw scan")
    if search_scan_cmd.yaw_rate_rad_s <= 0.0:
        failures.append("search should start a yaw scan after the settle window")
    if (
        abs(search_scan_cmd.roll_rate_rad_s) > 1e-9
        or abs(search_scan_cmd.pitch_rate_rad_s) > 1e-9
    ):
        failures.append("search scan should not command roll or pitch")
    if pitched_search_cmd.pitch_rate_rad_s <= 0.0:
        failures.append("pitched-down search should command positive pitch-rate to level")
    if abs(pitched_search_cmd.yaw_rate_rad_s) > 1e-9:
        failures.append("pitched search should level before starting yaw scan")
    if moving_search_cmd.pitch_rate_rad_s <= 0.0:
        failures.append("forward search drift should command positive pitch-rate braking")
    if moving_search_cmd.roll_rate_rad_s >= 0.0:
        failures.append("rightward search drift should command negative roll-rate braking")
    if abs(moving_search_cmd.yaw_rate_rad_s) > 1e-9:
        failures.append("moving search should brake before starting yaw scan")

    if failures:
        print("verdict=FAIL")
        for failure in failures:
            print(f"failure={failure}")
        return 1

    print("verdict=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

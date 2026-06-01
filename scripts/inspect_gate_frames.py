#!/usr/bin/env python3
"""Inspect gate detections and control commands on image/video frames.

This is a presentation/debug tool for the active VQ1 algorithm. It does not use
world pose, simulator gate IDs, depth, or map truth.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm import AutonomousRacingPilot, VehicleTelemetry
from algorithm.control_adapter import to_betaflight_rc_fields
from algorithm.gate_detector import CameraParams, GateDetector
from algorithm.gate_tracker import GateTracker
from algorithm.types import GateEstimate, RacingCommand, TrackMode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_hsv(raw: str) -> np.ndarray:
    parts = [int(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be comma-separated H,S,V")
    if not all(0 <= value <= 255 for value in parts):
        raise argparse.ArgumentTypeError("HSV values must be in [0, 255]")
    return np.array(parts, dtype=np.uint8)


def iter_image_paths(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def make_demo_frame(
    width: int = 640,
    height: int = 360,
    *,
    center: tuple[int, int] = (380, 162),
    outer: int = 132,
    inner: int | None = None,
    visible: bool = True,
) -> np.ndarray:
    frame = np.full((height, width, 3), (28, 34, 38), dtype=np.uint8)
    cv2.line(frame, (0, height - 44), (width, height - 98), (65, 70, 74), 2)
    cv2.line(frame, (0, height - 10), (width, height - 40), (50, 56, 60), 2)

    if not visible:
        return frame

    inner = inner if inner is not None else max(8, int(round(outer * 0.67)))
    color = (0, 165, 255)
    cv2.rectangle(
        frame,
        (center[0] - outer // 2, center[1] - outer // 2),
        (center[0] + outer // 2, center[1] + outer // 2),
        color,
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (center[0] - inner // 2, center[1] - inner // 2),
        (center[0] + inner // 2, center[1] + inner // 2),
        (28, 34, 38),
        thickness=-1,
    )
    return frame


def iter_demo_sequence(
    total_frames: int,
    fps: float,
) -> Iterator[tuple[int, float, np.ndarray, str]]:
    """Yield a small synthetic approach/loss sequence for presentation plots."""
    total_frames = max(1, total_frames)
    for index in range(total_frames):
        progress = min(1.0, index / max(1, total_frames * 0.58))
        center_x = int(round(458 + (326 - 458) * progress))
        center_y = int(round(152 + (180 - 152) * progress))
        outer = int(round(72 + (330 - 72) * progress))

        # After the approach, hide the gate long enough to show tracker memory
        # and eventual search fallback.
        visible = index < int(total_frames * 0.72)
        frame = make_demo_frame(center=(center_x, center_y), outer=outer, visible=visible)
        yield index, index / fps, frame, f"synthetic_{index:03d}"


def iter_frames(args: argparse.Namespace) -> Iterator[tuple[int, float, np.ndarray, str]]:
    if args.demo:
        if args.demo_frames > 1:
            yield from iter_demo_sequence(args.demo_frames, args.assumed_fps)
        else:
            yield 0, 0.0, make_demo_frame(), "synthetic_demo"
        return

    if args.source is None:
        raise ValueError("Provide --source or use --demo.")

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_dir() or source.suffix.lower() in IMAGE_EXTS:
        for index, path in enumerate(iter_image_paths(source)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"warning: could not read image {path}", file=sys.stderr)
                continue
            yield index, index / args.assumed_fps, frame, path.stem
        return

    if source.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported source type: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not np.isfinite(fps):
        fps = args.assumed_fps

    raw_index = 0
    kept_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_index % args.stride == 0:
                timestamp = raw_index / fps
                yield kept_index, timestamp, frame, f"frame_{raw_index:06d}"
                kept_index += 1
                if args.max_frames and kept_index >= args.max_frames:
                    break
            raw_index += 1
    finally:
        cap.release()


def mode_color(mode: TrackMode) -> tuple[int, int, int]:
    return {
        TrackMode.DETECTED: (90, 220, 90),
        TrackMode.TRACKED: (0, 220, 255),
        TrackMode.COMMIT: (255, 220, 80),
        TrackMode.SEARCH: (190, 190, 190),
        TrackMode.RECOVER: (80, 80, 255),
    }.get(mode, (220, 220, 220))


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    gate: GateEstimate,
    command: RacingCommand,
    rc_fields: dict[str, int],
    frame_id: int,
    timestamp_s: float,
) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    color = mode_color(gate.mode)

    cv2.drawMarker(
        overlay,
        (w // 2, h // 2),
        (220, 220, 220),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )

    if gate.pixel_center is not None:
        cx, cy = int(round(gate.pixel_center[0])), int(round(gate.pixel_center[1]))
        cv2.circle(overlay, (cx, cy), 7, color, thickness=2)
        cv2.line(overlay, (w // 2, h // 2), (cx, cy), color, thickness=2)

        if gate.apparent_size_px is not None:
            half = int(round(gate.apparent_size_px / 2.0))
            cv2.rectangle(
                overlay,
                (cx - half, cy - half),
                (cx + half, cy + half),
                color,
                thickness=2,
            )

    dist = "--" if gate.distance_m is None else f"{gate.distance_m:.2f}m"
    put_text(overlay, f"frame={frame_id} t={timestamp_s:.2f}s", (12, 22))
    put_text(
        overlay,
        (
            f"mode={gate.mode.value} conf={gate.confidence:.2f} "
            f"bearing=({gate.bearing_h_rad:+.2f},{gate.bearing_v_rad:+.2f}) "
            f"dist={dist}"
        ),
        (12, 44),
        color=color,
    )
    put_text(
        overlay,
        (
            f"cmd roll={command.roll_rate_rad_s:+.2f} "
            f"pitch={command.pitch_rate_rad_s:+.2f} "
            f"yaw={command.yaw_rate_rad_s:+.2f} thrust={command.thrust_norm:.2f}"
        ),
        (12, h - 38),
    )
    put_text(
        overlay,
        (
            f"rc thr={rc_fields['throttle']} roll={rc_fields['roll']} "
            f"pitch={rc_fields['pitch']} yaw={rc_fields['yaw']}"
        ),
        (12, h - 16),
    )
    return overlay


def write_row(
    writer: csv.DictWriter,
    frame_id: int,
    source_name: str,
    timestamp_s: float,
    gate: GateEstimate,
    command: RacingCommand,
    rc_fields: dict[str, int],
) -> None:
    writer.writerow(
        {
            "frame_id": frame_id,
            "source": source_name,
            "timestamp_s": f"{timestamp_s:.6f}",
            "mode": gate.mode.value,
            "confidence": f"{gate.confidence:.6f}",
            "bearing_h_rad": f"{gate.bearing_h_rad:.6f}",
            "bearing_v_rad": f"{gate.bearing_v_rad:.6f}",
            "distance_m": "" if gate.distance_m is None else f"{gate.distance_m:.6f}",
            "pixel_x": "" if gate.pixel_center is None else f"{gate.pixel_center[0]:.3f}",
            "pixel_y": "" if gate.pixel_center is None else f"{gate.pixel_center[1]:.3f}",
            "apparent_size_px": (
                "" if gate.apparent_size_px is None else f"{gate.apparent_size_px:.3f}"
            ),
            "gate_age_s": f"{gate.age_s:.6f}",
            "roll_rate_rad_s": f"{command.roll_rate_rad_s:.6f}",
            "pitch_rate_rad_s": f"{command.pitch_rate_rad_s:.6f}",
            "yaw_rate_rad_s": f"{command.yaw_rate_rad_s:.6f}",
            "thrust_norm": f"{command.thrust_norm:.6f}",
            "rc_throttle": rc_fields["throttle"],
            "rc_roll": rc_fields["roll"],
            "rc_pitch": rc_fields["pitch"],
            "rc_yaw": rc_fields["yaw"],
            "rc_arm": rc_fields["arm"],
        }
    )


def write_mask(
    frame: np.ndarray,
    out_path: Path,
    detector: GateDetector,
) -> None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detector.hsv_lo, detector.hsv_hi)
    cv2.imwrite(str(out_path), mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Image, directory, or video to inspect.")
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic gate frame.")
    parser.add_argument("--demo-frames", type=int, default=1, help="Synthetic demo length.")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/gate_inspection"))
    parser.add_argument("--hsv-lo", type=parse_hsv, default=parse_hsv("10,150,150"))
    parser.add_argument("--hsv-hi", type=parse_hsv, default=parse_hsv("35,255,255"))
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth video frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--assumed-fps", type=float, default=30.0)
    parser.add_argument("--save-mask", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if not args.demo and args.source is None:
        parser.error("Provide --source or --demo")

    out_dir = args.out_dir
    overlay_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    detector = GateDetector(
        hsv_lo=args.hsv_lo,
        hsv_hi=args.hsv_hi,
        min_contour_area=args.min_area,
    )
    tracker = GateTracker(detector=detector, camera_params=CameraParams())
    pilot = AutonomousRacingPilot(tracker=tracker, frame_format="bgr")
    telemetry = VehicleTelemetry()

    fields = [
        "frame_id",
        "source",
        "timestamp_s",
        "mode",
        "confidence",
        "bearing_h_rad",
        "bearing_v_rad",
        "distance_m",
        "pixel_x",
        "pixel_y",
        "apparent_size_px",
        "gate_age_s",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "thrust_norm",
        "rc_throttle",
        "rc_roll",
        "rc_pitch",
        "rc_yaw",
        "rc_arm",
    ]

    trace_path = out_dir / "trace.csv"
    count = 0
    detections = 0
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for frame_id, timestamp_s, frame, source_name in iter_frames(args):
            command, gate = pilot.update(frame, telemetry, timestamp_s=timestamp_s)
            rc_fields = to_betaflight_rc_fields(command)
            if gate.is_usable:
                detections += 1

            overlay = draw_overlay(frame, gate, command, rc_fields, frame_id, timestamp_s)
            safe_name = f"{frame_id:06d}_{source_name}".replace("/", "_")
            cv2.imwrite(str(overlay_dir / f"{safe_name}.jpg"), overlay)
            if args.save_mask:
                write_mask(frame, mask_dir / f"{safe_name}.png", detector)
            write_row(writer, frame_id, source_name, timestamp_s, gate, command, rc_fields)
            count += 1

            if args.max_frames and count >= args.max_frames:
                break

    print(f"processed_frames={count}")
    print(f"usable_gate_frames={detections}")
    print(f"trace={trace_path}")
    print(f"overlays={overlay_dir}")
    if args.save_mask:
        print(f"masks={mask_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

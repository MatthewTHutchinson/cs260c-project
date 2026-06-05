#!/usr/bin/env python3
"""Compare reactive and learned controllers on the same trace rows."""

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

import matplotlib.pyplot as plt
import numpy as np

from algorithm.learned_controller import LearnedFeatureController
from algorithm.reactive_controller import ReactiveGateController
from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry


COMMANDS = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
TEACHER_KEYS = (
    "teacher_roll_rate_rad_s",
    "teacher_pitch_rate_rad_s",
    "teacher_yaw_rate_rad_s",
    "teacher_thrust_norm",
)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def gate_from_row(row: dict[str, str]) -> GateEstimate:
    mode = row.get("mode", "detected")
    try:
        track_mode = TrackMode(mode)
    except ValueError:
        track_mode = TrackMode.DETECTED
    return GateEstimate(
        bearing_h_rad=as_float(row, "bearing_h_rad"),
        bearing_v_rad=as_float(row, "bearing_v_rad"),
        distance_m=as_float(row, "distance_m"),
        confidence=as_float(row, "confidence"),
        pixel_center=(as_float(row, "pixel_x"), as_float(row, "pixel_y")),
        apparent_size_px=as_float(row, "apparent_size_px"),
        sequence_index=int(as_float(row, "next_gate_index", 0.0)),
        age_s=as_float(row, "gate_age_s"),
        mode=track_mode,
    )


def telemetry_from_row(row: dict[str, str]) -> VehicleTelemetry:
    return VehicleTelemetry(
        rpy_rad=np.asarray([0.0, as_float(row, "body_forward_elevation_rad"), 0.0]),
        linear_velocity_m_s=np.asarray(
            [
                as_float(row, "body_vx_m_s"),
                as_float(row, "body_vy_m_s"),
                as_float(row, "body_vz_m_s"),
            ],
            dtype=np.float64,
        ),
        timestamp_s=as_float(row, "timestamp_s"),
    )


def command_array(command: RacingCommand) -> np.ndarray:
    return np.asarray(
        [
            command.roll_rate_rad_s,
            command.pitch_rate_rad_s,
            command.yaw_rate_rad_s,
            command.thrust_norm,
        ],
        dtype=np.float64,
    )


def load_rows(path: Path, course: str | None, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if course:
        rows = [row for row in rows if row.get("course") == course]
    rows = [row for row in rows if row.get("mode") in {TrackMode.DETECTED.value, TrackMode.COMMIT.value}]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no usable rows found in {path}")
    return rows


def summarize(label: str, pred: np.ndarray, target: np.ndarray) -> list[str]:
    error = pred - target
    mae = np.mean(np.abs(error), axis=0)
    mse = float(np.mean(error * error))
    return [
        (
            f"{label} samples={len(pred)} mse={mse:.8f} "
            + " ".join(f"mae_{name}={value:.6f}" for name, value in zip(COMMANDS, mae))
        )
    ]


def write_comparison(
    path: Path,
    rows: list[dict[str, str]],
    learned: np.ndarray,
    reactive: np.ndarray,
    teacher: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "course",
        "mode",
        "timestamp_s",
        *[f"teacher_{name}" for name in COMMANDS],
        *[f"learned_{name}" for name in COMMANDS],
        *[f"reactive_{name}" for name in COMMANDS],
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, y_l, y_r, y_t in zip(rows, learned, reactive, teacher):
            out = {
                "course": row.get("course", ""),
                "mode": row.get("mode", ""),
                "timestamp_s": row.get("timestamp_s", ""),
            }
            for prefix, values in (("teacher", y_t), ("learned", y_l), ("reactive", y_r)):
                for name, value in zip(COMMANDS, values):
                    out[f"{prefix}_{name}"] = f"{value:.8f}"
            writer.writerow(out)


def plot_comparison(path: Path, learned: np.ndarray, reactive: np.ndarray, teacher: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(teacher))
    fig, axes = plt.subplots(len(COMMANDS), 1, figsize=(11.0, 9.0), sharex=True)
    for idx, (ax, name) in enumerate(zip(axes, COMMANDS)):
        ax.plot(x, teacher[:, idx], label=f"teacher_{name}", color="#1971c2", linewidth=1.3)
        ax.plot(x, learned[:, idx], label=f"learned_{name}", color="#2b8a3e", linewidth=1.0)
        ax.plot(x, reactive[:, idx], label=f"reactive_{name}", color="#e67700", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(True, color="#d9dde3")
        ax.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("sample index")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/privileged_teacher/trace_with_variants.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--course", default="s_curve")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path("logs/controller_comparison/trace_comparison.csv"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.trace, args.course, args.limit)
    learned_controller = LearnedFeatureController(args.checkpoint)
    reactive_controller = ReactiveGateController()

    learned_commands: list[np.ndarray] = []
    reactive_commands: list[np.ndarray] = []
    teacher_commands: list[np.ndarray] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)

    for idx, row in enumerate(rows):
        gate = gate_from_row(row)
        telemetry = telemetry_from_row(row)
        learned_commands.append(command_array(learned_controller.compute(gate, telemetry)))
        reactive_commands.append(command_array(reactive_controller.compute(gate, telemetry)))
        teacher_commands.append(np.asarray([as_float(row, key) for key in TEACHER_KEYS], dtype=np.float64))
        grouped_indices[row.get("mode", "unknown") or "unknown"].append(idx)

    learned = np.vstack(learned_commands)
    reactive = np.vstack(reactive_commands)
    teacher = np.vstack(teacher_commands)
    print(f"trace={args.trace}")
    print(f"checkpoint={args.checkpoint}")
    print(f"course={args.course}")
    print(f"rows={len(rows)}")
    for line in summarize("learned_vs_teacher", learned, teacher):
        print(line)
    for line in summarize("reactive_vs_teacher", reactive, teacher):
        print(line)
    for line in summarize("learned_vs_reactive", learned, reactive):
        print(line)
    for mode, indices in sorted(grouped_indices.items()):
        idxs = np.asarray(indices, dtype=np.int64)
        for line in summarize(f"mode={mode} learned_vs_teacher", learned[idxs], teacher[idxs]):
            print(line)
        for line in summarize(f"mode={mode} reactive_vs_teacher", reactive[idxs], teacher[idxs]):
            print(line)

    write_comparison(args.out, rows, learned, reactive, teacher)
    print(f"comparison={args.out}")
    if args.plot:
        plot_comparison(args.plot, learned, reactive, teacher)
        print(f"plot={args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

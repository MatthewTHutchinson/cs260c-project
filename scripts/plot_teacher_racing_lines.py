#!/usr/bin/env python3
"""Plot privileged-teacher racing lines for report/debug inspection."""

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

import matplotlib.pyplot as plt
import numpy as np


BASE_COURSE_ORDER = (
    "easy",
    "lateral_soft",
    "low_high",
    "four_gate_straight",
    "circular_arc",
    "s_curve",
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return rows


def group_by_course(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("course", "unknown") or "unknown"].append(row)
    return dict(grouped)


def ordered_courses(grouped: dict[str, list[dict[str, str]]], *, include_random: bool) -> list[str]:
    base = [course for course in BASE_COURSE_ORDER if course in grouped]
    remaining = sorted(course for course in grouped if course not in BASE_COURSE_ORDER)
    if include_random:
        return base + remaining
    return base


def unique_gates(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    by_gate: dict[int, dict[str, float]] = {}
    for row in rows:
        raw_idx = row.get("teacher_next_gate_index", "")
        try:
            gate_idx = int(float(raw_idx))
        except ValueError:
            continue
        if gate_idx in by_gate:
            continue
        by_gate[gate_idx] = {
            "idx": float(gate_idx),
            "x": as_float(row, "teacher_next_gate_x_m"),
            "y": as_float(row, "teacher_next_gate_y_m"),
            "z": as_float(row, "teacher_next_gate_z_m"),
            "yaw_deg": as_float(row, "teacher_next_gate_yaw_deg", 0.0),
        }
    return [by_gate[idx] for idx in sorted(by_gate)]


def course_role(course: str, test_courses: set[str]) -> str:
    if course in test_courses:
        return "held-out test"
    if "_rand_" in course:
        return "train augment"
    return "train base"


def style_for_role(role: str) -> tuple[str, str]:
    if role == "held-out test":
        return "#c2255c", "#fff0f6"
    if role == "train augment":
        return "#1971c2", "#f8f9fa"
    return "#2b8a3e", "#f8f9fa"


def phase_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("teacher_phase", "unknown") or "unknown" for row in rows)
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def plot_course(ax: plt.Axes, course: str, rows: list[dict[str, str]], *, role: str) -> None:
    color, face = style_for_role(role)
    ax.set_facecolor(face)
    nominal = [row for row in rows if row.get("teacher_phase") == "nominal"]
    launch = [row for row in rows if row.get("teacher_phase") == "launch"]
    off_nominal = [row for row in rows if row.get("teacher_phase") == "off_nominal"]
    line_rows = nominal or rows

    world_x = np.asarray([as_float(row, "world_x_m") for row in line_rows], dtype=np.float64)
    world_y = np.asarray([as_float(row, "world_y_m") for row in line_rows], dtype=np.float64)
    target_x = np.asarray([as_float(row, "teacher_target_x_m") for row in line_rows], dtype=np.float64)
    target_y = np.asarray([as_float(row, "teacher_target_y_m") for row in line_rows], dtype=np.float64)

    ax.plot(world_x, world_y, color=color, linewidth=2.0, label="teacher line")
    ax.plot(
        target_x,
        target_y,
        color="#495057",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
        label="lookahead",
    )

    if launch:
        ax.scatter(
            [as_float(row, "world_x_m") for row in launch],
            [as_float(row, "world_y_m") for row in launch],
            color="#15aabf",
            s=7,
            alpha=0.45,
            label="launch",
        )
    if off_nominal:
        ax.scatter(
            [as_float(row, "world_x_m") for row in off_nominal],
            [as_float(row, "world_y_m") for row in off_nominal],
            color="#f08c00",
            s=5,
            alpha=0.20,
            label="off-nominal",
        )

    gates = unique_gates(rows)
    if gates:
        gate_x = [gate["x"] for gate in gates]
        gate_y = [gate["y"] for gate in gates]
        ax.scatter(gate_x, gate_y, color="#212529", marker="s", s=28, label="gates", zorder=4)
        for gate in gates:
            yaw = math.radians(gate["yaw_deg"])
            dx = 1.2 * math.cos(yaw)
            dy = 1.2 * math.sin(yaw)
            ax.arrow(
                gate["x"],
                gate["y"],
                dx,
                dy,
                width=0.018,
                head_width=0.22,
                head_length=0.30,
                length_includes_head=True,
                color="#343a40",
                alpha=0.70,
                zorder=5,
            )
            ax.text(
                gate["x"],
                gate["y"] + 0.32,
                str(int(gate["idx"])),
                fontsize=7,
                ha="center",
                color="#212529",
            )

    ax.set_title(f"{course}\n{role}", fontsize=9, fontweight="bold")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#d9dde3", linewidth=0.6)
    ax.axis("equal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("logs/privileged_teacher/trace_augmented_rejoin.csv"))
    parser.add_argument("--out", type=Path, default=Path("logs/privileged_teacher/teacher_racing_lines_base.png"))
    parser.add_argument("--summary-out", type=Path, default=Path("logs/privileged_teacher/teacher_racing_lines_summary.csv"))
    parser.add_argument("--include-random", action="store_true", help="Plot randomized augmentation courses too.")
    parser.add_argument("--test-courses", default="s_curve")
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args.trace)
    grouped = group_by_course(rows)
    test_courses = set(split_csv(args.test_courses))
    courses = ordered_courses(grouped, include_random=args.include_random)
    if not courses:
        raise ValueError("no courses selected for plotting")

    cols = max(1, args.cols)
    rows_n = int(math.ceil(len(courses) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.3 * cols, 3.5 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, course in zip(axes.ravel(), courses):
        ax.set_visible(True)
        role = course_role(course, test_courses)
        plot_course(ax, course, grouped[course], role=role)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(
        "Privileged Teacher Racing Lines",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("course", "role", "rows", "gates", "phase_counts"),
        )
        writer.writeheader()
        for course in courses:
            writer.writerow(
                {
                    "course": course,
                    "role": course_role(course, test_courses),
                    "rows": len(grouped[course]),
                    "gates": len(unique_gates(grouped[course])),
                    "phase_counts": phase_counts(grouped[course]),
                }
            )

    print(f"trace={args.trace}")
    print(f"courses_plotted={len(courses)}")
    print(f"test_courses={','.join(sorted(test_courses)) if test_courses else 'none'}")
    print(f"plot={args.out}")
    print(f"summary={args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit learned-policy prediction CSVs for calibration before rollout."""

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

import matplotlib.pyplot as plt
import numpy as np


COMMANDS = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
LIMITS = {
    "roll_rate": (-0.70, 0.70),
    "pitch_rate": (-0.80, 0.80),
    "yaw_rate": (-1.20, 1.20),
    "thrust": (0.30, 0.95),
}


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"prediction CSV has no rows: {path}")
    return rows


def arrays_for(rows: list[dict[str, str]], command: str) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray([as_float(row, f"pred_{command}") for row in rows], dtype=np.float64)
    target = np.asarray([as_float(row, f"target_{command}") for row in rows], dtype=np.float64)
    return pred, target


def pct(values: np.ndarray, mask: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(mask[finite]) * 100.0)


def summarize(label: str, rows: list[dict[str, str]]) -> list[str]:
    out = [f"{label} rows={len(rows)}"]
    for command in COMMANDS:
        pred, target = arrays_for(rows, command)
        error = pred - target
        lo, hi = LIMITS[command]
        out.append(
            (
                f"{label} {command} "
                f"pred_range=[{np.nanmin(pred):.4f},{np.nanmax(pred):.4f}] "
                f"target_range=[{np.nanmin(target):.4f},{np.nanmax(target):.4f}] "
                f"bias={np.nanmean(error):.6f} "
                f"mae={np.nanmean(np.abs(error)):.6f} "
                f"pred_clip_pct={pct(pred, (pred <= lo + 1e-6) | (pred >= hi - 1e-6)):.1f} "
                f"target_clip_pct={pct(target, (target <= lo + 1e-6) | (target >= hi - 1e-6)):.1f}"
            )
        )
    return out


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "unknown") or "unknown"].append(row)
    return dict(sorted(grouped.items()))


def print_worst_rows(rows: list[dict[str, str]], limit: int) -> None:
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        total = 0.0
        for command in COMMANDS:
            total += as_float(row, f"abs_error_{command}", 0.0)
        scored.append((total, row))
    for rank, (score, row) in enumerate(sorted(scored, reverse=True)[:limit], start=1):
        print(
            "worst_row "
            f"rank={rank} score={score:.6f} "
            f"course={row.get('course', '')} mode={row.get('mode', '')} "
            f"timestamp_s={row.get('timestamp_s', '')}"
        )


def plot_predictions(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(rows), dtype=np.float64)
    fig, axes = plt.subplots(len(COMMANDS), 1, figsize=(11.0, 9.0), sharex=True)
    for ax, command in zip(axes, COMMANDS):
        pred, target = arrays_for(rows, command)
        ax.plot(t, target, label=f"target_{command}", color="#1971c2", linewidth=1.2)
        ax.plot(t, pred, label=f"pred_{command}", color="#e67700", linewidth=1.0, alpha=0.85)
        lo, hi = LIMITS[command]
        ax.axhline(lo, color="#adb5bd", linewidth=0.8)
        ax.axhline(hi, color="#adb5bd", linewidth=0.8)
        ax.set_ylabel(command)
        ax.grid(True, color="#d9dde3")
        ax.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("sample index")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    print(f"predictions={args.predictions}")
    for line in summarize("overall", rows):
        print(line)
    for course, course_rows in group_by(rows, "course").items():
        for line in summarize(f"course={course}", course_rows):
            print(line)
    for mode, mode_rows in group_by(rows, "mode").items():
        for line in summarize(f"mode={mode}", mode_rows):
            print(line)
    print_worst_rows(rows, args.worst)
    if args.plot:
        plot_predictions(rows, args.plot)
        print(f"plot={args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

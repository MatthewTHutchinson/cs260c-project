#!/usr/bin/env python3
"""Plot detector/tracker/controller traces from the active racing algorithm."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODE_TO_CODE = {
    "search": 0,
    "tracked": 1,
    "detected": 2,
    "commit": 3,
    "recover": -1,
}


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def load_trace(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"trace has no rows: {path}")

    t = np.array([as_float(row, "timestamp_s", i) for i, row in enumerate(rows)])
    mode = np.array([MODE_TO_CODE.get(row.get("mode", ""), np.nan) for row in rows])

    return {
        "t": t,
        "mode": mode,
        "confidence": np.array([as_float(row, "confidence") for row in rows]),
        "bearing_h": np.array([as_float(row, "bearing_h_rad") for row in rows]),
        "bearing_v": np.array([as_float(row, "bearing_v_rad") for row in rows]),
        "distance": np.array([as_float(row, "distance_m") for row in rows]),
        "roll": np.array([as_float(row, "roll_rate_rad_s") for row in rows]),
        "pitch": np.array([as_float(row, "pitch_rate_rad_s") for row in rows]),
        "yaw": np.array([as_float(row, "yaw_rate_rad_s") for row in rows]),
        "thrust": np.array([as_float(row, "thrust_norm") for row in rows]),
        "rc_throttle": np.array([as_float(row, "rc_throttle") for row in rows]),
        "rc_roll": np.array([as_float(row, "rc_roll") for row in rows]),
        "rc_pitch": np.array([as_float(row, "rc_pitch") for row in rows]),
        "rc_yaw": np.array([as_float(row, "rc_yaw") for row in rows]),
    }


def style_axis(ax) -> None:
    ax.grid(True, color="#d9dde3", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trace(trace: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    t = trace["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.0), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0].step(t, trace["mode"], where="post", color="#3b5bdb", linewidth=2.0)
    axes[0].set_ylabel("mode")
    axes[0].set_yticks([-1, 0, 1, 2, 3])
    axes[0].set_yticklabels(["recover", "search", "tracked", "detected", "commit"])
    axes[0].set_ylim(-1.4, 3.4)
    style_axis(axes[0])
    conf_axis = axes[0].twinx()
    conf_axis.plot(t, trace["confidence"], color="#2f9e44", linewidth=1.8, label="confidence")
    conf_axis.set_ylabel("confidence")
    conf_axis.set_ylim(-0.05, 1.05)
    conf_axis.spines["top"].set_visible(False)
    conf_axis.legend(loc="upper right", frameon=False)

    axes[1].plot(t, trace["bearing_h"], label="horizontal", color="#e67700", linewidth=1.8)
    axes[1].plot(t, trace["bearing_v"], label="vertical", color="#0b7285", linewidth=1.8)
    axes[1].axhline(0.0, color="#868e96", linewidth=1.0)
    axes[1].set_ylabel("bearing rad")
    axes[1].legend(loc="upper right", frameon=False)
    style_axis(axes[1])

    axes[2].plot(t, trace["roll"], label="roll", color="#5f3dc4", linewidth=1.6)
    axes[2].plot(t, trace["pitch"], label="pitch", color="#c2255c", linewidth=1.6)
    axes[2].plot(t, trace["yaw"], label="yaw", color="#1971c2", linewidth=1.6)
    axes[2].plot(t, trace["thrust"], label="thrust", color="#2b8a3e", linewidth=1.6)
    axes[2].axhline(0.0, color="#868e96", linewidth=1.0)
    axes[2].set_ylabel("command")
    axes[2].legend(loc="upper right", ncol=4, frameon=False)
    style_axis(axes[2])

    axes[3].plot(t, trace["rc_throttle"], label="throttle", color="#2b8a3e", linewidth=1.4)
    axes[3].plot(t, trace["rc_roll"], label="roll", color="#5f3dc4", linewidth=1.4)
    axes[3].plot(t, trace["rc_pitch"], label="pitch", color="#c2255c", linewidth=1.4)
    axes[3].plot(t, trace["rc_yaw"], label="yaw", color="#1971c2", linewidth=1.4)
    axes[3].set_ylabel("RC PWM")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", ncol=4, frameon=False)
    style_axis(axes[3])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Autonomous Racing Pilot Trace")
    args = parser.parse_args()

    trace = load_trace(args.trace)
    plot_trace(trace, args.out, args.title)
    print(f"plot={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

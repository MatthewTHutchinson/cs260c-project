#!/usr/bin/env python3
"""Evaluate a feature policy checkpoint on trace CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from learning.datasets import NormalizedDataset, TraceSequenceDataset
from learning.datasets import FeatureSpec
from learning.feature_policy import load_checkpoint


COMMAND_NAMES = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
COMMAND_LIMITS = (0.70, 0.80, 1.20, 0.90)


def summarize_group(label: str, pred: np.ndarray, target: np.ndarray) -> list[str]:
    if len(pred) == 0:
        return []
    error = pred - target
    mae = np.mean(np.abs(error), axis=0)
    mse = float(np.mean(error * error))
    lines = [
        (
            f"{label} samples={len(pred)} mse={mse:.8f} "
            + " ".join(f"mae_{name}={value:.6f}" for name, value in zip(COMMAND_NAMES, mae))
        )
    ]
    saturation_parts = []
    for idx, (name, limit) in enumerate(zip(COMMAND_NAMES, COMMAND_LIMITS)):
        if name == "thrust":
            pred_sat = np.mean((pred[:, idx] <= 0.35 + 1e-6) | (pred[:, idx] >= 0.90 - 1e-6))
            target_sat = np.mean((target[:, idx] <= 0.35 + 1e-6) | (target[:, idx] >= 0.90 - 1e-6))
        else:
            pred_sat = np.mean(np.abs(pred[:, idx]) >= limit - 1e-6)
            target_sat = np.mean(np.abs(target[:, idx]) >= limit - 1e-6)
        saturation_parts.append(f"{name}_pred={pred_sat * 100.0:.1f}")
        saturation_parts.append(f"{name}_target={target_sat * 100.0:.1f}")
    lines.append(f"{label} saturation_pct " + " ".join(saturation_parts))
    return lines


def write_predictions(
    path: Path,
    *,
    rows: list[dict[str, str]],
    pred: np.ndarray,
    target: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "trace_path",
        "course",
        "mode",
        "timestamp_s",
        "pred_roll_rate",
        "pred_pitch_rate",
        "pred_yaw_rate",
        "pred_thrust",
        "target_roll_rate",
        "target_pitch_rate",
        "target_yaw_rate",
        "target_thrust",
        "abs_error_roll_rate",
        "abs_error_pitch_rate",
        "abs_error_yaw_rate",
        "abs_error_thrust",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, y_hat, y in zip(rows, pred, target):
            abs_error = np.abs(y_hat - y)
            writer.writerow(
                {
                    "trace_path": row.get("_trace_path", ""),
                    "course": row.get("course", ""),
                    "mode": row.get("mode", ""),
                    "timestamp_s": row.get("timestamp_s", ""),
                    "pred_roll_rate": f"{y_hat[0]:.8f}",
                    "pred_pitch_rate": f"{y_hat[1]:.8f}",
                    "pred_yaw_rate": f"{y_hat[2]:.8f}",
                    "pred_thrust": f"{y_hat[3]:.8f}",
                    "target_roll_rate": f"{y[0]:.8f}",
                    "target_pitch_rate": f"{y[1]:.8f}",
                    "target_yaw_rate": f"{y[2]:.8f}",
                    "target_thrust": f"{y[3]:.8f}",
                    "abs_error_roll_rate": f"{abs_error[0]:.8f}",
                    "abs_error_pitch_rate": f"{abs_error[1]:.8f}",
                    "abs_error_yaw_rate": f"{abs_error[2]:.8f}",
                    "abs_error_thrust": f"{abs_error[3]:.8f}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--include-courses", nargs="*", help="Only evaluate these course names.")
    parser.add_argument("--exclude-courses", nargs="*", help="Skip these course names.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(str(args.checkpoint), map_location=device)
    model.to(device)
    sequence_length = int(payload.get("metadata", {}).get("sequence_length", 12))
    spec = FeatureSpec(
        feature_names=tuple(payload.get("feature_names", FeatureSpec.default().feature_names))
    )
    dataset = TraceSequenceDataset(
        args.traces,
        sequence_length=sequence_length,
        spec=spec,
        include_courses=args.include_courses,
        exclude_courses=args.exclude_courses,
    )
    data = NormalizedDataset(dataset, payload["feature_mean"], payload["feature_std"])
    loader = DataLoader(data, batch_size=args.batch_size)
    loss_fn = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_count = 0
    abs_error = torch.zeros(4, dtype=torch.float64, device=device)
    pred_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total_loss += float(loss_fn(pred, y).item())
            abs_error += (pred - y).abs().sum(dim=0).double()
            total_count += len(x)
            pred_batches.append(pred.detach().cpu())
            target_batches.append(y.detach().cpu())

    pred_np = torch.cat(pred_batches, dim=0).numpy()
    target_np = torch.cat(target_batches, dim=0).numpy()
    mae = (abs_error / max(1, total_count)).detach().cpu().tolist()
    print(f"samples={total_count}")
    print(f"mse={total_loss / max(1, total_count * 4):.8f}")
    for name, value in zip(COMMAND_NAMES, mae):
        print(f"mae_{name}={value:.8f}")

    by_course: dict[str, list[int]] = defaultdict(list)
    by_mode: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset.sample_rows):
        by_course[row.get("course", "unknown") or "unknown"].append(idx)
        by_mode[row.get("mode", "unknown") or "unknown"].append(idx)

    for course in sorted(by_course):
        idxs = np.asarray(by_course[course], dtype=np.int64)
        for line in summarize_group(f"course={course}", pred_np[idxs], target_np[idxs]):
            print(line)

    for mode in sorted(by_mode):
        idxs = np.asarray(by_mode[mode], dtype=np.int64)
        for line in summarize_group(f"mode={mode}", pred_np[idxs], target_np[idxs]):
            print(line)

    if args.predictions_out:
        write_predictions(
            args.predictions_out,
            rows=dataset.sample_rows,
            pred=pred_np,
            target=target_np,
        )
        print(f"predictions={args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a feature policy checkpoint on trace CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from learning.datasets import NormalizedDataset, TraceSequenceDataset
from learning.feature_policy import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(str(args.checkpoint), map_location=device)
    model.to(device)
    sequence_length = int(payload.get("metadata", {}).get("sequence_length", 12))
    dataset = TraceSequenceDataset(args.traces, sequence_length=sequence_length)
    data = NormalizedDataset(dataset, payload["feature_mean"], payload["feature_std"])
    loader = DataLoader(data, batch_size=args.batch_size)
    loss_fn = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_count = 0
    abs_error = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total_loss += float(loss_fn(pred, y).item())
            abs_error += (pred - y).abs().sum(dim=0).double()
            total_count += len(x)

    names = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
    mae = (abs_error / max(1, total_count)).detach().cpu().tolist()
    print(f"samples={total_count}")
    print(f"mse={total_loss / max(1, total_count * 4):.8f}")
    for name, value in zip(names, mae):
        print(f"mae_{name}={value:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


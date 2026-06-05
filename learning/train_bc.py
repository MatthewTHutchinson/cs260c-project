#!/usr/bin/env python3
"""Train a feature-based behavioral cloning policy."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from learning.datasets import (
    NormalizedDataset,
    TraceSequenceDataset,
    fit_feature_normalizer,
    make_synthetic_trace,
)
from learning.feature_policy import FeaturePolicyGRU, PolicyConfig, save_checkpoint


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = max(1, int(n * val_fraction)) if n > 1 else 0
    return indices[val_n:], indices[:val_n]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model(x), y)
            total += float(loss.item()) * len(x)
            count += len(x)
    return total / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="*", default=[])
    parser.add_argument("--demo-synthetic", action="store_true")
    parser.add_argument("--synthetic-out", type=Path, default=Path("logs/learning_synthetic/trace.csv"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/feature_bc.pt"))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-courses", nargs="*", help="Only train on these course names.")
    parser.add_argument("--exclude-courses", nargs="*", help="Skip these course names.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    traces = list(args.traces)
    if args.demo_synthetic:
        traces.append(make_synthetic_trace(args.synthetic_out))
    if not traces:
        raise SystemExit("provide --traces or use --demo-synthetic")

    dataset = TraceSequenceDataset(
        traces,
        sequence_length=args.sequence_length,
        stride=args.stride,
        include_courses=args.include_courses,
        exclude_courses=args.exclude_courses,
    )
    train_idx, val_idx = split_indices(len(dataset), args.val_fraction, args.seed)
    train_base = Subset(dataset, train_idx)
    val_base = Subset(dataset, val_idx) if val_idx else train_base
    mean, std = fit_feature_normalizer(train_base)
    train_data = NormalizedDataset(train_base, mean, std)
    val_data = NormalizedDataset(val_base, mean, std)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeaturePolicyGRU(
        PolicyConfig(input_dim=dataset.input_dim, hidden_dim=args.hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(x)
            count += len(x)
        train_loss /= max(1, count)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        print(f"epoch={epoch:03d} train_mse={train_loss:.8f} val_mse={val_loss:.8f}")
        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(
                str(args.out),
                model=model,
                feature_mean=mean,
                feature_std=std,
                feature_names=dataset.feature_names,
                metadata={
                    "sequence_length": args.sequence_length,
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "best_val_mse": best_val,
                    "include_courses": args.include_courses or [],
                    "exclude_courses": args.exclude_courses or [],
                },
            )

    print(f"checkpoint={args.out}")
    print(f"best_val_mse={best_val:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

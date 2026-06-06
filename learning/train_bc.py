#!/usr/bin/env python3
"""Train a feature-based behavioral cloning policy."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

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


def parse_weight_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    weights: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected name=weight in {raw!r}, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty weight key in {raw!r}")
        weight = float(value)
        if weight <= 0.0:
            raise ValueError(f"weight for {key!r} must be positive")
        weights[key] = weight
    return weights


def sample_weight_for_row(
    row: dict[str, str],
    *,
    phase_weights: dict[str, float],
    mode_weights: dict[str, float],
    episode_weights: dict[str, float],
    command_source_weights: dict[str, float],
    trace_weights: dict[str, float],
) -> float:
    phase = str(row.get("teacher_phase", "")).strip()
    mode = str(row.get("mode", "")).strip()
    episode = str(row.get("episode_id", "")).strip()
    command_source = str(row.get("command_source", "")).strip()
    trace_path = str(row.get("_trace_path", "")).strip()
    trace_weight = 1.0
    if trace_path:
        path = Path(trace_path)
        for key in (trace_path, path.name, path.stem):
            if key in trace_weights:
                trace_weight = trace_weights[key]
                break
    return (
        phase_weights.get(phase, 1.0)
        * mode_weights.get(mode, 1.0)
        * episode_weights.get(episode, 1.0)
        * command_source_weights.get(command_source, 1.0)
        * trace_weight
    )


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
    parser.add_argument(
        "--no-prev-command-features",
        action="store_true",
        help="Drop previous command inputs for runtime-friendly closed-loop inference.",
    )
    parser.add_argument(
        "--no-sequence-features",
        action="store_true",
        help=(
            "Drop last_gate_passed and next_gate_index inputs so the policy cannot "
            "lean on perfect teacher-side gate sequence labels."
        ),
    )
    parser.add_argument(
        "--phase-sampling-weights",
        default="",
        help="Comma-separated teacher_phase=weight values, e.g. off_nominal=4,launch=2.",
    )
    parser.add_argument(
        "--mode-sampling-weights",
        default="",
        help="Comma-separated mode=weight values, e.g. commit=2,detected=1.",
    )
    parser.add_argument(
        "--episode-sampling-weights",
        default="",
        help="Comma-separated episode_id=weight values for source-aware DAgger balancing.",
    )
    parser.add_argument(
        "--command-source-sampling-weights",
        default="",
        help="Comma-separated command_source=weight values, e.g. reactive_fallback=2,learned=1.",
    )
    parser.add_argument(
        "--trace-sampling-weights",
        default="",
        help="Comma-separated trace filename/stem/path=weight values for relabel selection.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    traces = list(args.traces)
    if args.demo_synthetic:
        traces.append(make_synthetic_trace(args.synthetic_out))
    if not traces:
        raise SystemExit("provide --traces or use --demo-synthetic")

    from learning.datasets import FeatureSpec

    spec = FeatureSpec.default(
        include_prev_command=not args.no_prev_command_features,
        include_sequence_features=not args.no_sequence_features,
    )

    dataset = TraceSequenceDataset(
        traces,
        sequence_length=args.sequence_length,
        stride=args.stride,
        spec=spec,
        include_courses=args.include_courses,
        exclude_courses=args.exclude_courses,
    )
    train_idx, val_idx = split_indices(len(dataset), args.val_fraction, args.seed)
    train_base = Subset(dataset, train_idx)
    val_base = Subset(dataset, val_idx) if val_idx else train_base
    mean, std = fit_feature_normalizer(train_base)
    train_data = NormalizedDataset(train_base, mean, std)
    val_data = NormalizedDataset(val_base, mean, std)
    phase_weights = parse_weight_map(args.phase_sampling_weights)
    mode_weights = parse_weight_map(args.mode_sampling_weights)
    episode_weights = parse_weight_map(args.episode_sampling_weights)
    command_source_weights = parse_weight_map(args.command_source_sampling_weights)
    trace_weights = parse_weight_map(args.trace_sampling_weights)
    sampler = None
    if (
        phase_weights
        or mode_weights
        or episode_weights
        or command_source_weights
        or trace_weights
    ):
        sample_weights = torch.as_tensor(
            [
                sample_weight_for_row(
                    dataset.sample_rows[index],
                    phase_weights=phase_weights,
                    mode_weights=mode_weights,
                    episode_weights=episode_weights,
                    command_source_weights=command_source_weights,
                    trace_weights=trace_weights,
                )
                for index in train_idx
            ],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(
            "weighted_sampling="
            f"phase={phase_weights or '{}'} "
            f"mode={mode_weights or '{}'} "
            f"episode={episode_weights or '{}'} "
            f"command_source={command_source_weights or '{}'} "
            f"trace={trace_weights or '{}'}"
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
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
                    "no_prev_command_features": bool(args.no_prev_command_features),
                    "no_sequence_features": bool(args.no_sequence_features),
                    "phase_sampling_weights": phase_weights,
                    "mode_sampling_weights": mode_weights,
                    "episode_sampling_weights": episode_weights,
                    "command_source_sampling_weights": command_source_weights,
                    "trace_sampling_weights": trace_weights,
                },
            )

    print(f"checkpoint={args.out}")
    print(f"best_val_mse={best_val:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a feature-policy `.pt` checkpoint to a NumPy runtime `.npz` file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def tensor_to_numpy(state: dict[str, torch.Tensor], key: str) -> np.ndarray:
    return state[key].detach().cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["model_state"]
    metadata = payload.get("metadata", {})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        feature_names=np.asarray(payload["feature_names"], dtype=str),
        sequence_length=np.asarray(int(metadata.get("sequence_length", 12)), dtype=np.int64),
        feature_mean=payload["feature_mean"].detach().cpu().numpy().astype(np.float32),
        feature_std=payload["feature_std"].detach().cpu().numpy().astype(np.float32),
        gru_weight_ih=tensor_to_numpy(state, "gru.weight_ih_l0"),
        gru_weight_hh=tensor_to_numpy(state, "gru.weight_hh_l0"),
        gru_bias_ih=tensor_to_numpy(state, "gru.bias_ih_l0"),
        gru_bias_hh=tensor_to_numpy(state, "gru.bias_hh_l0"),
        ln_weight=tensor_to_numpy(state, "head.0.weight"),
        ln_bias=tensor_to_numpy(state, "head.0.bias"),
        head1_weight=tensor_to_numpy(state, "head.1.weight"),
        head1_bias=tensor_to_numpy(state, "head.1.bias"),
        head2_weight=tensor_to_numpy(state, "head.3.weight"),
        head2_bias=tensor_to_numpy(state, "head.3.bias"),
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"npz={args.out}")
    print(f"sequence_length={int(metadata.get('sequence_length', 12))}")
    print(f"features={len(payload['feature_names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

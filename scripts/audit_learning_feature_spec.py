#!/usr/bin/env python3
"""Audit learned-policy feature columns for privileged or brittle inputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
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

import numpy as np

from learning.datasets import FeatureSpec, SEQUENCE_FEATURE_COLUMNS, TARGET_COLUMNS


PRIVILEGED_PREFIXES = (
    "debug_world_",
    "world_",
    "teacher_",
    "reference_",
)
PRIVILEGED_EXACT = {
    "gate_id",
    "gate_index",
    "gate_center_x_m",
    "gate_center_y_m",
    "gate_center_z_m",
}


def load_checkpoint_features(path: Path) -> tuple[str, ...]:
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        return tuple(str(name) for name in payload["feature_names"].tolist())
    import torch

    payload = torch.load(path, map_location="cpu")
    return tuple(str(name) for name in payload["feature_names"])


def trace_columns(path: Path) -> tuple[str, ...]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return tuple(reader.fieldnames or ())


def is_privileged(name: str) -> bool:
    return name in PRIVILEGED_EXACT or any(name.startswith(prefix) for prefix in PRIVILEGED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional .pt or .npz policy checkpoint to inspect.")
    parser.add_argument("--trace", type=Path, help="Optional trace CSV to compare against selected features.")
    parser.add_argument("--no-prev-command-features", action="store_true")
    parser.add_argument("--no-sequence-features", action="store_true")
    parser.add_argument("--expect-no-prev-command-features", action="store_true")
    parser.add_argument("--expect-no-sequence-features", action="store_true")
    parser.add_argument(
        "--allow-missing-selected-features",
        action="store_true",
        help="Warn instead of failing if a selected non-derived feature is absent from the trace.",
    )
    args = parser.parse_args()

    if args.checkpoint:
        feature_names = load_checkpoint_features(args.checkpoint)
        source = str(args.checkpoint)
    else:
        feature_names = FeatureSpec.default(
            include_prev_command=not args.no_prev_command_features,
            include_sequence_features=not args.no_sequence_features,
        ).feature_names
        source = "constructed_feature_spec"

    feature_set = set(feature_names)
    privileged_features = sorted(name for name in feature_names if is_privileged(name))
    sequence_features = sorted(name for name in SEQUENCE_FEATURE_COLUMNS if name in feature_set)
    prev_command_features = sorted(f"prev_{name}" for name in TARGET_COLUMNS if f"prev_{name}" in feature_set)
    failures: list[str] = []

    if privileged_features:
        failures.append(f"privileged_features_present={','.join(privileged_features)}")
    if args.expect_no_sequence_features and sequence_features:
        failures.append(f"sequence_features_present={','.join(sequence_features)}")
    if args.expect_no_prev_command_features and prev_command_features:
        failures.append(f"prev_command_features_present={','.join(prev_command_features)}")

    print(f"source={source}")
    print(f"feature_count={len(feature_names)}")
    print(f"sequence_features={','.join(sequence_features) if sequence_features else 'none'}")
    print(f"prev_command_features={','.join(prev_command_features) if prev_command_features else 'none'}")
    print(f"privileged_features={','.join(privileged_features) if privileged_features else 'none'}")

    if args.trace:
        columns = trace_columns(args.trace)
        selected_missing = sorted(name for name in feature_names if name not in columns and not name.startswith("mode_") and not name.endswith("_delta") and name != "has_distance")
        privileged_columns = sorted(name for name in columns if is_privileged(name))
        print(f"trace={args.trace}")
        print(f"trace_columns={len(columns)}")
        print(f"trace_privileged_columns={','.join(privileged_columns) if privileged_columns else 'none'}")
        print(f"selected_trace_columns_missing={','.join(selected_missing) if selected_missing else 'none'}")
        if selected_missing and not args.allow_missing_selected_features:
            failures.append(f"selected_trace_columns_missing={','.join(selected_missing)}")

    if failures:
        print("verdict=FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("verdict=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

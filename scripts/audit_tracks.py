#!/usr/bin/env python3
"""Static audit for track geometry, config splits, and train/test leakage."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_track_library() -> dict[str, list[dict]]:
    """Load env/tracks.py without importing env/__init__.py and PyBullet."""
    tracks_path = ROOT_DIR / "env" / "tracks.py"
    spec = importlib.util.spec_from_file_location("track_audit_tracks", tracks_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load track module from {tracks_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRACK_LIBRARY


TRACK_LIBRARY = _load_track_library()


TRAIN_PREFIXES = ("rect_", "train_")
TEST_PREFIXES = ("heldout_", "audit_")


def _centers(track: list[dict]) -> np.ndarray:
    return np.array([gate["center"] for gate in track], dtype=np.float64)


def _track_metrics(name: str, track: list[dict]) -> dict:
    centers = _centers(track)
    segments = np.roll(centers, -1, axis=0) - centers
    segment_lengths = np.linalg.norm(segments, axis=1)
    dz = np.abs(segments[:, 2])
    x = centers[:, 0]
    y = centers[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    turns = []
    for idx in range(len(centers)):
        incoming = (centers[idx] - centers[(idx - 1) % len(centers)])[:2]
        outgoing = (centers[(idx + 1) % len(centers)] - centers[idx])[:2]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm < 1e-9 or outgoing_norm < 1e-9:
            continue
        dot = float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
        turns.append(float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))))

    radii = [float(gate["radius"]) for gate in track]
    return {
        "name": name,
        "gate_count": len(track),
        "direction": "ccw" if signed_area > 0.0 else "cw",
        "signed_area_xy": signed_area,
        "total_length_m": float(segment_lengths.sum()),
        "min_segment_m": float(segment_lengths.min()),
        "max_segment_m": float(segment_lengths.max()),
        "z_min_m": float(centers[:, 2].min()),
        "z_max_m": float(centers[:, 2].max()),
        "z_range_m": float(np.ptp(centers[:, 2])),
        "max_dz_m": float(dz.max()),
        "min_turn_deg": float(min(turns)) if turns else 0.0,
        "max_turn_deg": float(max(turns)) if turns else 0.0,
        "mean_turn_deg": float(np.mean(turns)) if turns else 0.0,
        "radius_min_m": float(min(radii)),
        "radius_max_m": float(max(radii)),
    }


def _aligned_rmsd(track_a: list[dict], track_b: list[dict]) -> float | None:
    a = _centers(track_a)
    b = _centers(track_b)
    if len(a) != len(b):
        return None

    best = float("inf")
    for reverse in (False, True):
        candidate = b[::-1] if reverse else b
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, shift, axis=0)
            rmsd = float(np.sqrt(np.mean(np.sum((a - shifted) ** 2, axis=1))))
            best = min(best, rmsd)
    return best


def _nearest_training_analogs() -> list[dict]:
    train_names = [name for name in TRACK_LIBRARY if name.startswith(TRAIN_PREFIXES)]
    test_names = [name for name in TRACK_LIBRARY if name.startswith(TEST_PREFIXES)]
    rows = []
    for test_name in test_names:
        candidates = []
        for train_name in train_names:
            rmsd = _aligned_rmsd(TRACK_LIBRARY[test_name], TRACK_LIBRARY[train_name])
            if rmsd is not None:
                candidates.append((rmsd, train_name))
        rmsd, train_name = min(candidates, key=lambda item: item[0])
        rows.append(
            {
                "track_name": test_name,
                "nearest_train_like_track": train_name,
                "aligned_rmsd_m": rmsd,
            }
        )
    return rows


def _config_track_sets(config_path: Path) -> list[tuple[str, list[str]]]:
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)

    sets: list[tuple[str, list[str]]] = []
    if isinstance(cfg, dict) and "env" in cfg:
        sets.append(("env", list(cfg.get("env", {}).get("track_names", []))))
    for idx, suite in enumerate(cfg.get("validation_suites", []) if isinstance(cfg, dict) else []):
        suite_name = str(suite.get("name", f"suite_{idx + 1}"))
        sets.append(
            (
                f"validation_suites[{idx}].{suite_name}",
                list(suite.get("env", {}).get("track_names", [])),
            )
        )
    if isinstance(cfg, dict) and "validation_env" in cfg:
        sets.append(("validation_env", list(cfg.get("validation_env", {}).get("track_names", []))))
    return sets


def _config_audit_rows(config_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        for label, track_names in _config_track_sets(path):
            if not track_names:
                continue
            unknown = [name for name in track_names if name not in TRACK_LIBRARY]
            known = [name for name in track_names if name in TRACK_LIBRARY]
            gate_counts = Counter(len(TRACK_LIBRARY[name]) for name in known)
            duplicates = sorted(name for name, count in Counter(track_names).items() if count > 1)
            rows.append(
                {
                    "config": str(path),
                    "set": label,
                    "n_tracks": len(track_names),
                    "gate_counts": json.dumps(dict(sorted(gate_counts.items()))),
                    "duplicates": json.dumps(duplicates),
                    "unknown_tracks": json.dumps(unknown),
                    "mixed_gate_counts": len(gate_counts) > 1,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, metrics: list[dict], analogs: list[dict], config_rows: list[dict]) -> None:
    gate_counts = Counter(row["gate_count"] for row in metrics)
    directions = Counter(row["direction"] for row in metrics)
    close_analogs = [row for row in analogs if row["aligned_rmsd_m"] < 0.35]
    mixed_configs = [row for row in config_rows if row["mixed_gate_counts"]]
    duplicated_configs = [row for row in config_rows if json.loads(row["duplicates"])]

    lines = [
        "# Track Audit",
        "",
        "## Summary",
        "",
        f"- Track count: `{len(metrics)}`",
        f"- Gate counts: `{dict(sorted(gate_counts.items()))}`",
        f"- Directions: `{dict(sorted(directions.items()))}`",
        f"- Held-out/audit tracks within `0.35 m` RMSD of a train-like track: `{len(close_analogs)}`",
        f"- Config track sets with mixed gate counts: `{len(mixed_configs)}`",
        f"- Config track sets using duplicate track names as weighting: `{len(duplicated_configs)}`",
        "",
        "## Closest Held-Out/Audit Analogs",
        "",
        "| Track | Nearest train-like track | RMSD m |",
        "| --- | --- | ---: |",
    ]
    for row in sorted(analogs, key=lambda item: item["aligned_rmsd_m"]):
        lines.append(
            f"| {row['track_name']} | {row['nearest_train_like_track']} | {row['aligned_rmsd_m']:.3f} |"
        )

    lines.extend(["", "## Config Issues", "", "| Config | Set | Gate counts | Duplicates |", "| --- | --- | --- | --- |"])
    for row in config_rows:
        duplicates = json.loads(row["duplicates"])
        if not row["mixed_gate_counts"] and not duplicates and json.loads(row["unknown_tracks"]) == []:
            continue
        lines.append(
            f"| {row['config']} | {row['set']} | {row['gate_counts']} | {', '.join(duplicates)} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--out", default="logs/track_audit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [_track_metrics(name, track) for name, track in TRACK_LIBRARY.items()]
    analogs = _nearest_training_analogs()
    config_rows = _config_audit_rows(Path(args.config_dir))

    _write_csv(out_dir / "track_metrics.csv", metrics)
    _write_csv(out_dir / "nearest_training_analogs.csv", analogs)
    _write_csv(out_dir / "config_track_audit.csv", config_rows)
    _write_report(out_dir / "report.md", metrics, analogs, config_rows)

    summary = {
        "track_count": len(metrics),
        "gate_counts": dict(sorted(Counter(row["gate_count"] for row in metrics).items())),
        "directions": dict(sorted(Counter(row["direction"] for row in metrics).items())),
        "close_heldout_or_audit_analogs": sum(row["aligned_rmsd_m"] < 0.35 for row in analogs),
        "mixed_gate_count_config_sets": sum(row["mixed_gate_counts"] for row in config_rows),
        "duplicate_weighted_config_sets": sum(bool(json.loads(row["duplicates"])) for row in config_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote audit artifacts to {out_dir}")


if __name__ == "__main__":
    main()

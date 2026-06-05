"""Datasets for feature-based behavioral cloning.

The first learned policy intentionally consumes detector/tracker features and
allowed telemetry, not raw images or simulator truth. CSV columns are optional
so this loader can use both full simulator traces and smaller inspection traces.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


MODE_NAMES = ("search", "tracked", "detected", "commit", "recover")
TARGET_COLUMNS = (
    "roll_rate_rad_s",
    "pitch_rate_rad_s",
    "yaw_rate_rad_s",
    "thrust_norm",
)
TEACHER_TARGET_COLUMNS = (
    "teacher_roll_rate_rad_s",
    "teacher_pitch_rate_rad_s",
    "teacher_yaw_rate_rad_s",
    "teacher_thrust_norm",
)

BASE_FEATURE_COLUMNS = (
    "frame_fresh",
    "last_gate_passed",
    "next_gate_index",
    "body_forward_elevation_rad",
    "body_vx_m_s",
    "body_vy_m_s",
    "body_vz_m_s",
    "confidence",
    "bearing_h_rad",
    "bearing_v_rad",
    "distance_m",
    "pixel_x",
    "pixel_y",
    "apparent_size_px",
    "gate_age_s",
)


def _as_float(raw: object, default: float = 0.0) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if text == "":
            return default
        try:
            value = float(text)
        except ValueError:
            return default
    if not math.isfinite(value):
        return default
    return value


def read_trace_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"trace has no rows: {path}")
    return rows


def discover_trace_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.rglob("trace.csv")))
        elif path.exists():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise ValueError("no trace CSV files found")
    return out


@dataclass(frozen=True)
class FeatureSpec:
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...] = TARGET_COLUMNS
    prefer_teacher_targets: bool = True

    @classmethod
    def default(cls) -> "FeatureSpec":
        names: list[str] = list(BASE_FEATURE_COLUMNS)
        names.extend(f"mode_{mode}" for mode in MODE_NAMES)
        names.extend(f"prev_{name}" for name in TARGET_COLUMNS)
        names.extend(
            (
                "bearing_h_delta",
                "bearing_v_delta",
                "distance_delta",
                "has_distance",
            )
        )
        return cls(feature_names=tuple(names))


def rows_to_feature_arrays(
    rows: list[dict[str, str]],
    spec: FeatureSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    spec = spec or FeatureSpec.default()
    features: list[list[float]] = []
    targets: list[list[float]] = []
    prev_target = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    prev_bearing_h = 0.0
    prev_bearing_v = 0.0
    prev_distance = 0.0

    for row in rows:
        target_names = spec.target_names
        if spec.prefer_teacher_targets and all(name in row for name in TEACHER_TARGET_COLUMNS):
            target_names = TEACHER_TARGET_COLUMNS
        target = np.array([_as_float(row.get(name)) for name in target_names], dtype=np.float32)
        mode = str(row.get("mode", "")).strip().lower()
        distance = _as_float(row.get("distance_m"))
        bearing_h = _as_float(row.get("bearing_h_rad"))
        bearing_v = _as_float(row.get("bearing_v_rad"))
        values: dict[str, float] = {name: _as_float(row.get(name)) for name in BASE_FEATURE_COLUMNS}
        values["has_distance"] = 1.0 if distance > 0.0 else 0.0
        values["bearing_h_delta"] = bearing_h - prev_bearing_h
        values["bearing_v_delta"] = bearing_v - prev_bearing_v
        values["distance_delta"] = distance - prev_distance
        for mode_name in MODE_NAMES:
            values[f"mode_{mode_name}"] = 1.0 if mode == mode_name else 0.0
        for name, value in zip(TARGET_COLUMNS, prev_target):
            values[f"prev_{name}"] = float(value)

        features.append([values.get(name, 0.0) for name in spec.feature_names])
        targets.append(target.tolist())
        prev_target = target
        prev_bearing_h = bearing_h
        prev_bearing_v = bearing_v
        prev_distance = distance

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        spec.feature_names,
    )


class TraceSequenceDataset(Dataset):
    """Windowed sequence dataset; each sample predicts the command at the end."""

    def __init__(
        self,
        traces: Iterable[Path],
        *,
        sequence_length: int = 12,
        stride: int = 1,
        spec: FeatureSpec | None = None,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        trace_paths = discover_trace_paths(traces)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        feature_names: tuple[str, ...] | None = None
        for trace in trace_paths:
            rows = read_trace_csv(trace)
            features, targets, names = rows_to_feature_arrays(rows, spec)
            feature_names = names
            if len(features) < self.sequence_length:
                continue
            for start in range(0, len(features) - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                xs.append(features[start:end])
                ys.append(targets[end - 1])

        if not xs:
            raise ValueError("no sequence samples were created from the provided traces")
        self.x = torch.from_numpy(np.stack(xs).astype(np.float32))
        self.y = torch.from_numpy(np.stack(ys).astype(np.float32))
        self.feature_names = feature_names or FeatureSpec.default().feature_names

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[-1])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class NormalizedDataset(Dataset):
    def __init__(self, base: Dataset, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = base
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return (x - self.mean) / self.std, y


def fit_feature_normalizer(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        xs.append(x)
    stacked = torch.stack(xs)
    return stacked.mean(dim=(0, 1)), stacked.std(dim=(0, 1))


def make_synthetic_trace(path: Path, *, rows: int = 240) -> Path:
    """Create a tiny trace that is easy for the policy to overfit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "timestamp_s",
        "frame_fresh",
        "last_gate_passed",
        "next_gate_index",
        "body_forward_elevation_rad",
        "body_vx_m_s",
        "body_vy_m_s",
        "body_vz_m_s",
        "mode",
        "confidence",
        "bearing_h_rad",
        "bearing_v_rad",
        "distance_m",
        "pixel_x",
        "pixel_y",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "thrust_norm",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(rows):
            t = i / 30.0
            bearing_h = 0.45 * math.sin(t * 1.7)
            bearing_v = 0.18 * math.cos(t * 1.1)
            distance = max(0.5, 8.5 - 0.035 * i)
            confidence = 0.35 + 0.6 * (0.5 + 0.5 * math.cos(t * 0.7))
            mode = "commit" if distance < 2.0 else "detected"
            writer.writerow(
                {
                    "timestamp_s": f"{t:.6f}",
                    "frame_fresh": int(i % 2 == 0),
                    "last_gate_passed": int(i // 90) - 1,
                    "next_gate_index": int(i // 90),
                    "body_forward_elevation_rad": f"{0.04 * math.sin(t):.6f}",
                    "body_vx_m_s": f"{1.2 + 0.2 * math.sin(t):.6f}",
                    "body_vy_m_s": f"{0.2 * math.sin(t * 0.5):.6f}",
                    "body_vz_m_s": f"{0.1 * math.cos(t * 0.5):.6f}",
                    "mode": mode,
                    "confidence": f"{confidence:.6f}",
                    "bearing_h_rad": f"{bearing_h:.6f}",
                    "bearing_v_rad": f"{bearing_v:.6f}",
                    "distance_m": f"{distance:.6f}",
                    "pixel_x": f"{320 + 320 * bearing_h:.3f}",
                    "pixel_y": f"{180 - 250 * bearing_v:.3f}",
                    "roll_rate_rad_s": f"{0.28 * bearing_h:.6f}",
                    "pitch_rate_rad_s": f"{-0.06 * distance + 0.02 * bearing_v:.6f}",
                    "yaw_rate_rad_s": f"{1.7 * bearing_h:.6f}",
                    "thrust_norm": f"{0.55 + 0.02 * bearing_v:.6f}",
                }
            )
    return path

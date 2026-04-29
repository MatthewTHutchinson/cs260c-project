"""Named gate layouts for training and held-out evaluation."""

from __future__ import annotations

import numpy as np


def _normalize_xy(vec: np.ndarray) -> np.ndarray:
    out = np.array([vec[0], vec[1], 0.0], dtype=np.float64)
    norm = np.linalg.norm(out[:2])
    if norm < 1e-9:
        raise ValueError("Track tangent is degenerate; gate centers are too close together.")
    return out / norm


def track_from_centers(
    centers: list[list[float]] | np.ndarray,
    radius: float = 0.75,
) -> list[dict]:
    """Create a cyclic gate layout and infer each gate normal from the local track tangent."""
    pts = [np.array(c, dtype=np.float64) for c in centers]
    gates = []
    n = len(pts)
    for i, center in enumerate(pts):
        prev_center = pts[(i - 1) % n]
        next_center = pts[(i + 1) % n]
        tangent = next_center - prev_center
        normal = _normalize_xy(tangent)
        gates.append(
            dict(
                center=center.copy(),
                normal=normal,
                radius=float(radius),
            )
        )
    return gates


TRACK_LIBRARY: dict[str, list[dict]] = {
    "rect_default": track_from_centers([
        [2.5, 0.0, 1.5],
        [5.0, 2.5, 1.5],
        [2.5, 5.0, 1.5],
        [0.0, 2.5, 1.5],
    ]),
    "rect_narrow": track_from_centers([
        [2.5, 0.2, 1.5],
        [4.6, 2.3, 1.4],
        [2.4, 4.6, 1.7],
        [0.2, 2.4, 1.6],
    ], radius=0.70),
    "rect_wide": track_from_centers([
        [2.2, 0.0, 1.4],
        [5.6, 2.4, 1.6],
        [2.4, 5.2, 1.7],
        [-0.4, 2.6, 1.5],
    ]),
    "rect_tall": track_from_centers([
        [2.1, -0.1, 1.5],
        [4.8, 2.9, 1.4],
        [2.0, 6.1, 1.8],
        [-0.5, 2.8, 1.6],
    ]),
    "rect_skew": track_from_centers([
        [2.8, 0.2, 1.4],
        [5.2, 2.2, 1.8],
        [2.4, 4.9, 1.6],
        [-0.2, 2.7, 1.3],
    ]),
    "rect_offset": track_from_centers([
        [2.9, -0.4, 1.6],
        [5.5, 2.5, 1.2],
        [2.1, 5.3, 1.9],
        [-0.7, 2.6, 1.4],
    ]),
    "rect_drop": track_from_centers([
        [2.1, 0.1, 1.9],
        [4.9, 2.4, 1.2],
        [2.7, 5.6, 1.8],
        [-0.2, 2.4, 1.1],
    ], radius=0.72),
    "rect_bow": track_from_centers([
        [2.4, -0.3, 1.5],
        [5.4, 2.0, 1.8],
        [2.8, 5.2, 1.4],
        [-0.3, 3.0, 1.7],
    ], radius=0.72),
    "rect_fast": track_from_centers([
        [2.0, -0.2, 1.3],
        [5.8, 2.4, 1.5],
        [2.9, 5.8, 1.7],
        [-0.6, 2.8, 1.4],
    ], radius=0.78),
    "heldout_diamond": track_from_centers([
        [2.4, -0.2, 1.5],
        [5.0, 2.0, 1.8],
        [2.6, 4.7, 1.4],
        [0.1, 2.3, 1.7],
    ]),
    "heldout_tilted": track_from_centers([
        [2.0, 0.4, 1.3],
        [4.6, 2.7, 1.9],
        [2.1, 5.4, 1.5],
        [-0.6, 2.9, 1.6],
    ]),
    "heldout_pinched": track_from_centers([
        [2.5, -0.4, 1.4],
        [4.9, 1.9, 1.8],
        [2.7, 4.5, 1.2],
        [0.4, 2.4, 1.9],
    ], radius=0.64),
    "heldout_zigzag": track_from_centers([
        [2.0, 0.5, 1.2],
        [5.3, 2.2, 1.9],
        [1.9, 5.2, 1.4],
        [-0.8, 2.6, 1.8],
    ], radius=0.66),
    "heldout_lowhigh": track_from_centers([
        [2.3, -0.2, 1.1],
        [5.2, 2.6, 2.0],
        [2.5, 5.1, 1.2],
        [-0.4, 2.7, 1.9],
    ], radius=0.65),
}


def get_track(name: str) -> list[dict]:
    if name not in TRACK_LIBRARY:
        raise KeyError(f"Unknown track name: {name}")
    return [
        dict(
            center=g["center"].copy(),
            normal=g["normal"].copy(),
            radius=float(g["radius"]),
        )
        for g in TRACK_LIBRARY[name]
    ]


def get_tracks(names: list[str]) -> list[list[dict]]:
    return [get_track(name) for name in names]

"""Runtime detector selection for the competition-facing autopilot."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import numpy as np

from algorithm.camera_profiles import get_camera_profile
from algorithm.gate_detector import CameraParams, GateDetector
from algorithm.gate_tracker import GateDetectorBackend, GateTracker
from algorithm.neural_gate_detector import (
    GateNetONNXDetector,
    NeuralGateDetectorConfig,
    NeuralOutputFormat,
    NeuralPixelOutputSpace,
    OpenCVDNNGateDetector,
)


DetectorKind = Literal["classical", "onnx", "gatenet"]


@dataclass(frozen=True)
class DetectorFactoryConfig:
    """Configuration for detector construction.

    `gatenet` is intentionally an ONNX-backed runtime mode. The project does
    not vendor a GateNet checkout or weights; a model export is supplied at
    runtime through `model_path`.
    """

    kind: DetectorKind = "classical"
    model_path: str | None = None
    output_format: NeuralOutputFormat = "corners8"
    input_size: tuple[int, int] = (320, 180)
    gate_physical_width_m: float = 2.7
    min_confidence: float = 0.35
    normalized_output: bool = True
    pixel_output_space: NeuralPixelOutputSpace = "input"
    camera_profile: str | None = None
    hsv_lo: np.ndarray | None = None
    hsv_hi: np.ndarray | None = None
    min_contour_area: int = 200


def detector_config_from_env(
    *,
    kind: str | None = None,
    model_path: str | None = None,
    output_format: str | None = None,
    input_size: tuple[int, int] | None = None,
    gate_physical_width_m: float | None = None,
    min_confidence: float | None = None,
    normalized_output: bool | None = None,
    pixel_output_space: str | None = None,
    camera_profile: str | None = None,
    hsv_lo: np.ndarray | None = None,
    hsv_hi: np.ndarray | None = None,
    min_contour_area: int | None = None,
) -> DetectorFactoryConfig:
    """Build detector config from explicit values plus environment defaults."""

    selected_kind = normalize_detector_kind(
        kind
        or _env("CS260C_GATE_DETECTOR", "GATE_DETECTOR")
        or "classical"
    )
    selected_model = (
        model_path
        or _env(
            "CS260C_GATE_DETECTOR_MODEL",
            "GATE_DETECTOR_MODEL",
            "GATENET_MODEL_PATH",
        )
    )
    selected_output = _normalize_output_format(
        output_format
        or _env(
            "CS260C_GATE_DETECTOR_OUTPUT",
            "GATE_DETECTOR_OUTPUT",
            "GATENET_OUTPUT_FORMAT",
        )
        or "corners8"
    )
    selected_input_size = input_size or _input_size_from_env() or (320, 180)
    selected_gate_width = (
        gate_physical_width_m
        if gate_physical_width_m is not None
        else _env_float("CS260C_GATE_WIDTH_M", "GATE_WIDTH_M", default=2.7)
    )
    selected_min_conf = (
        min_confidence
        if min_confidence is not None
        else _env_float(
            "CS260C_GATE_DETECTOR_CONFIDENCE",
            "GATE_DETECTOR_CONFIDENCE",
            "GATENET_CONFIDENCE",
            default=0.35,
        )
    )
    selected_normalized = (
        normalized_output
        if normalized_output is not None
        else _env_bool(
            "CS260C_GATE_DETECTOR_NORMALIZED",
            "GATE_DETECTOR_NORMALIZED",
            "GATENET_NORMALIZED_OUTPUT",
            default=True,
        )
    )
    selected_pixel_space = _normalize_pixel_output_space(
        pixel_output_space
        or _env(
            "CS260C_GATE_DETECTOR_PIXEL_SPACE",
            "GATE_DETECTOR_PIXEL_SPACE",
            "GATENET_PIXEL_SPACE",
        )
        or "input"
    )
    selected_camera_profile = (
        camera_profile
        or _env("CS260C_CAMERA_PROFILE", "ELODIN_CAMERA_PROFILE")
        or None
    )
    selected_min_area = (
        min_contour_area
        if min_contour_area is not None
        else int(_env_float("CS260C_GATE_MIN_AREA", "GATE_MIN_AREA", default=200))
    )

    return DetectorFactoryConfig(
        kind=selected_kind,
        model_path=selected_model,
        output_format=selected_output,
        input_size=selected_input_size,
        gate_physical_width_m=selected_gate_width,
        min_confidence=selected_min_conf,
        normalized_output=selected_normalized,
        pixel_output_space=selected_pixel_space,
        camera_profile=selected_camera_profile,
        hsv_lo=hsv_lo,
        hsv_hi=hsv_hi,
        min_contour_area=selected_min_area,
    )


def build_gate_detector(config: DetectorFactoryConfig | None = None) -> GateDetectorBackend:
    """Construct the configured detector backend."""

    config = config or detector_config_from_env()
    kind = normalize_detector_kind(config.kind)
    if kind == "classical":
        return GateDetector(
            gate_physical_width=config.gate_physical_width_m,
            hsv_lo=config.hsv_lo,
            hsv_hi=config.hsv_hi,
            min_contour_area=config.min_contour_area,
        )

    if not config.model_path:
        raise ValueError(
            f"{kind} detector selected but no model path was provided. "
            "Set CS260C_GATE_DETECTOR_MODEL=/path/to/model.onnx or pass --model-path."
        )

    neural_config = NeuralGateDetectorConfig(
        model_path=config.model_path,
        output_format=config.output_format,
        input_size=config.input_size,
        normalized_output=config.normalized_output,
        pixel_output_space=config.pixel_output_space,
        gate_physical_width_m=config.gate_physical_width_m,
        min_confidence=config.min_confidence,
        source_name=kind,
    )
    if kind == "gatenet":
        return GateNetONNXDetector(neural_config)
    return OpenCVDNNGateDetector(neural_config)


def build_gate_tracker(
    config: DetectorFactoryConfig | None = None,
    *,
    camera_params: CameraParams | None = None,
) -> GateTracker:
    """Construct a `GateTracker` with the configured detector and camera model."""

    config = config or detector_config_from_env()
    profile = get_camera_profile(config.camera_profile)
    return GateTracker(
        detector=build_gate_detector(config),
        camera_params=camera_params or profile.camera_params(),
    )


def normalize_detector_kind(raw: str) -> DetectorKind:
    key = raw.strip().lower().replace("-", "_")
    aliases = {
        "classic": "classical",
        "hsv": "classical",
        "opencv": "onnx",
        "opencv_dnn": "onnx",
        "neural": "onnx",
    }
    key = aliases.get(key, key)
    if key not in {"classical", "onnx", "gatenet"}:
        raise ValueError("detector must be one of: classical, onnx, gatenet")
    return key  # type: ignore[return-value]


def parse_input_size(raw: str) -> tuple[int, int]:
    """Parse a detector input size as `WIDTHxHEIGHT`."""

    clean = raw.strip().lower().replace(",", "x")
    parts = [part for part in clean.split("x") if part]
    if len(parts) != 2:
        raise ValueError("input size must be WIDTHxHEIGHT, for example 320x180")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("input width and height must be positive")
    return width, height


def _input_size_from_env() -> tuple[int, int] | None:
    raw = _env(
        "CS260C_GATE_DETECTOR_INPUT_SIZE",
        "GATE_DETECTOR_INPUT_SIZE",
        "GATENET_INPUT_SIZE",
    )
    if raw:
        return parse_input_size(raw)

    width = _env(
        "CS260C_GATE_DETECTOR_INPUT_WIDTH",
        "GATE_DETECTOR_INPUT_WIDTH",
        "GATENET_INPUT_WIDTH",
    )
    height = _env(
        "CS260C_GATE_DETECTOR_INPUT_HEIGHT",
        "GATE_DETECTOR_INPUT_HEIGHT",
        "GATENET_INPUT_HEIGHT",
    )
    if width or height:
        if not width or not height:
            raise ValueError("detector input width and height must be provided together")
        return int(width), int(height)
    return None


def _normalize_output_format(raw: str) -> NeuralOutputFormat:
    key = raw.strip().lower().replace("-", "_")
    aliases = {
        "corners": "corners8",
        "corner": "corners8",
        "box": "bbox",
        "center_range": "center_distance",
        "center_dist": "center_distance",
        "mask": "heatmap",
        "segmentation": "heatmap",
    }
    key = aliases.get(key, key)
    if key not in {"corners8", "bbox", "center_distance", "heatmap"}:
        raise ValueError(
            "detector output must be one of: corners8, bbox, center_distance, heatmap"
        )
    return key  # type: ignore[return-value]


def _normalize_pixel_output_space(raw: str) -> NeuralPixelOutputSpace:
    key = raw.strip().lower().replace("-", "_")
    if key not in {"input", "frame"}:
        raise ValueError("pixel output space must be input or frame")
    return key  # type: ignore[return-value]


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_float(*names: str, default: float) -> float:
    value = _env(*names)
    return default if value is None else float(value)


def _env_bool(*names: str, default: bool) -> bool:
    value = _env(*names)
    if value is None:
        return default
    key = value.strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean env value for {names[0]}, got {value!r}")

"""Optional neural gate detector backend.

The active pilot consumes `GateObservation` objects, so learned gate perception
should plug in here without changing tracking, navigation, or command output.
This module intentionally uses OpenCV DNN for ONNX inference so it adds no hard
PyTorch/TensorFlow dependency to the default VQ1 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from algorithm.gate_detector import CameraParams, GateObservation

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


NeuralOutputFormat = Literal["corners8", "bbox", "center_distance", "heatmap"]
NeuralPixelOutputSpace = Literal["input", "frame"]


@dataclass(frozen=True)
class NeuralGateDetectorConfig:
    """Runtime settings for an exported neural gate detector."""

    model_path: str
    output_format: NeuralOutputFormat
    input_size: tuple[int, int] = (320, 180)
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0 / 255.0
    swap_rb: bool = True
    normalized_output: bool = True
    pixel_output_space: NeuralPixelOutputSpace = "input"
    gate_physical_width_m: float = 2.7
    min_confidence: float = 0.35
    mask_threshold: float = 0.5
    min_mask_area_px: float = 40.0
    max_forward_output_rows: int = 256
    source_name: str = "neural"


class OpenCVDNNGateDetector:
    """Convert an exported ONNX gate model into `GateObservation` candidates.

    Supported output formats:

    - `corners8`: rows of `tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y, conf`
    - `bbox`: rows of `cx, cy, width, height, conf...`, YOLO-like extra class
      scores are accepted and the maximum value after index 4 is used
    - `center_distance`: rows of `cx, cy, distance_m, conf`
    - `heatmap`: single-channel segmentation/confidence map

    Model-specific wrappers can subclass this if a research repo uses a custom
    tensor layout. The rest of the autonomy stack should not care.
    """

    def __init__(self, config: NeuralGateDetectorConfig) -> None:
        if not _CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for OpenCVDNNGateDetector. "
                "Install with: pip install opencv-python"
            )

        model_path = Path(config.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Neural gate model not found: {model_path}")

        self.config = config
        self.gate_w = config.gate_physical_width_m
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(
        self,
        bgr_frame: np.ndarray,
        cam: Optional[CameraParams] = None,
        max_gates: int = 2,
    ) -> list[GateObservation]:
        cam = cam or CameraParams()
        output = self._forward(bgr_frame)

        if self.config.output_format == "heatmap":
            return self._detections_from_heatmap(output, bgr_frame.shape, cam, max_gates)

        rows = self._as_rows(output)
        detections: list[GateObservation] = []
        for row in rows[: self.config.max_forward_output_rows]:
            obs = self._row_to_observation(row, cam)
            if obs is not None and obs.confidence >= self.config.min_confidence:
                detections.append(obs)

        detections.sort(key=lambda gate: gate.distance_est)
        return detections[:max_gates]

    def _forward(self, bgr_frame: np.ndarray) -> np.ndarray:
        width, height = self.config.input_size
        blob = cv2.dnn.blobFromImage(
            bgr_frame,
            scalefactor=self.config.scale,
            size=(width, height),
            mean=self.config.mean,
            swapRB=self.config.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        return self.net.forward()

    def _as_rows(self, output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            rows = arr[0]
            if rows.shape[0] <= 16 and rows.shape[1] > rows.shape[0]:
                return rows.T
            return rows
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            return arr.reshape(arr.shape[2], arr.shape[3])
        raise ValueError(f"Unsupported neural detector output shape: {arr.shape}")

    def _row_to_observation(
        self,
        row: np.ndarray,
        cam: CameraParams,
    ) -> Optional[GateObservation]:
        fmt = self.config.output_format
        if fmt == "corners8":
            if row.size < 9:
                return None
            coords = self._scale_points(row[:8].reshape(4, 2), cam)
            confidence = float(row[8])
            cx, cy = np.mean(coords, axis=0)
            top = np.linalg.norm(coords[1] - coords[0])
            bottom = np.linalg.norm(coords[2] - coords[3])
            left = np.linalg.norm(coords[3] - coords[0])
            right = np.linalg.norm(coords[2] - coords[1])
            apparent_size = max((top + bottom) * 0.5, (left + right) * 0.5)
            return self._make_observation(cx, cy, apparent_size, confidence, cam, coords)

        if fmt == "bbox":
            if row.size < 5:
                return None
            cx, cy, width, height = self._scale_box(row[:4], cam)
            confidence = float(np.max(row[4:])) if row.size > 5 else float(row[4])
            apparent_size = max(width, height)
            return self._make_observation(cx, cy, apparent_size, confidence, cam)

        if fmt == "center_distance":
            if row.size < 4:
                return None
            cx, cy = self._scale_point(row[:2], cam)
            distance_m = float(row[2])
            confidence = float(row[-1])
            bearing_h = float(np.arctan2(cx - cam.cx, cam.fx))
            bearing_v = float(np.arctan2(cam.cy - cy, cam.fy))
            return GateObservation(
                bearing_h=bearing_h,
                bearing_v=bearing_v,
                distance_est=distance_m,
                confidence=confidence,
                pixel_centre=(float(cx), float(cy)),
                source=self.config.source_name,
            )

        raise ValueError(f"Unsupported neural detector output format: {fmt}")

    def _detections_from_heatmap(
        self,
        output: np.ndarray,
        frame_shape: tuple[int, ...],
        cam: CameraParams,
        max_gates: int,
    ) -> list[GateObservation]:
        heatmap = np.asarray(output, dtype=np.float32).squeeze()
        if heatmap.ndim != 2:
            raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

        height, width = frame_shape[:2]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = (heatmap >= self.config.mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_mask_area_px:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_px, h_px), _ = rect
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [cnt], contourIdx=-1, color=255, thickness=-1)
            confidence = float(np.clip(np.mean(heatmap[contour_mask > 0]), 0.0, 1.0))
            detections.append(
                self._make_observation(cx, cy, max(w_px, h_px), confidence, cam)
            )

        detections.sort(key=lambda gate: gate.distance_est)
        return [
            gate
            for gate in detections
            if gate.confidence >= self.config.min_confidence
        ][:max_gates]

    def _make_observation(
        self,
        cx: float,
        cy: float,
        apparent_size_px: float,
        confidence: float,
        cam: CameraParams,
        corners: Optional[np.ndarray] = None,
    ) -> Optional[GateObservation]:
        if apparent_size_px < 2:
            return None

        bearing_h = float(np.arctan2(cx - cam.cx, cam.fx))
        bearing_v = float(np.arctan2(cam.cy - cy, cam.fy))
        distance = float(self.gate_w * cam.fx / (apparent_size_px + 1e-6))
        corners_px = None
        if corners is not None:
            corners_px = tuple((float(x), float(y)) for x, y in corners)

        return GateObservation(
            bearing_h=bearing_h,
            bearing_v=bearing_v,
            distance_est=distance,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            pixel_centre=(float(cx), float(cy)),
            corners_px=corners_px,
            source=self.config.source_name,
        )

    def _scale_point(self, point: np.ndarray, cam: CameraParams) -> tuple[float, float]:
        x, y = point.astype(float)
        if self.config.normalized_output:
            return float(x * cam.width), float(y * cam.height)
        if self.config.pixel_output_space == "input":
            input_w, input_h = self.config.input_size
            return float(x * cam.width / input_w), float(y * cam.height / input_h)
        return float(x), float(y)

    def _scale_points(self, points: np.ndarray, cam: CameraParams) -> np.ndarray:
        pts = points.astype(float).copy()
        if self.config.normalized_output:
            pts[:, 0] *= cam.width
            pts[:, 1] *= cam.height
        elif self.config.pixel_output_space == "input":
            input_w, input_h = self.config.input_size
            pts[:, 0] *= cam.width / input_w
            pts[:, 1] *= cam.height / input_h
        return pts

    def _scale_box(self, box: np.ndarray, cam: CameraParams) -> tuple[float, float, float, float]:
        cx, cy, width, height = box.astype(float)
        if self.config.normalized_output:
            return (
                float(cx * cam.width),
                float(cy * cam.height),
                float(width * cam.width),
                float(height * cam.height),
            )
        if self.config.pixel_output_space == "input":
            input_w, input_h = self.config.input_size
            return (
                float(cx * cam.width / input_w),
                float(cy * cam.height / input_h),
                float(width * cam.width / input_w),
                float(height * cam.height / input_h),
            )
        return float(cx), float(cy), float(width), float(height)


class GateNetONNXDetector(OpenCVDNNGateDetector):
    """Named wrapper for a GateNet-style model exported to ONNX.

    GateNet forks/exports can use different output heads. Keep the same
    `NeuralGateDetectorConfig.output_format` switch as the generic ONNX
    backend, but label observations as `gatenet` in debug traces.
    """

    def __init__(self, config: NeuralGateDetectorConfig) -> None:
        super().__init__(replace(config, source_name="gatenet"))

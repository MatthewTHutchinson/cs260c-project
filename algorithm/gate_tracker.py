"""Temporal gate tracker built around interchangeable gate detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from algorithm.gate_detector import CameraParams, GateDetector, GateObservation
from algorithm.types import GateEstimate, TrackMode


class GateDetectorBackend(Protocol):
    gate_w: float

    def detect(
        self,
        bgr_frame: np.ndarray,
        cam: CameraParams,
        max_gates: int = 2,
    ) -> list[GateObservation]:
        ...


@dataclass
class GateTrackerConfig:
    """Tuning knobs for short-horizon temporal tracking."""

    max_track_age_s: float = 0.85
    min_detect_confidence: float = 0.08
    tracked_confidence_decay: float = 0.55
    commit_distance_m: float = 2.2


class GateTracker:
    """Convert per-frame detections into a stable next-gate estimate."""

    def __init__(
        self,
        detector: Optional[GateDetectorBackend] = None,
        camera_params: Optional[CameraParams] = None,
        config: Optional[GateTrackerConfig] = None,
    ) -> None:
        self.detector = detector or GateDetector()
        self.camera_params = camera_params or CameraParams()
        self.config = config or GateTrackerConfig()
        self._last: Optional[GateEstimate] = None
        self._last_timestamp_s: Optional[float] = None

    def reset(self) -> None:
        self._last = None
        self._last_timestamp_s = None

    def update(
        self,
        frame: Optional[np.ndarray],
        timestamp_s: float,
        frame_format: str = "bgr",
    ) -> GateEstimate:
        """Return the best gate estimate from the current frame.

        `frame_format` is `"bgr"` for OpenCV frames and `"rgb"` for RGBA/RGB
        simulator frames. The detector itself expects BGR.
        """
        if frame is not None:
            detections = self._detect(frame, frame_format=frame_format)
            if detections:
                estimate = self._estimate_from_detection(detections[0], timestamp_s)
                self._last = estimate
                self._last_timestamp_s = timestamp_s
                return estimate

        return self._fallback(timestamp_s)

    def _detect(self, frame: np.ndarray, frame_format: str) -> list[GateObservation]:
        fmt = frame_format.strip().lower()
        if fmt == "rgb":
            frame_bgr = frame[:, :, ::-1]
        elif fmt == "bgr":
            frame_bgr = frame
        else:
            raise ValueError("frame_format must be 'bgr' or 'rgb'.")

        detections = self.detector.detect(frame_bgr, cam=self.camera_params, max_gates=2)
        return [
            det
            for det in detections
            if det.confidence >= self.config.min_detect_confidence
        ]

    def _estimate_from_detection(
        self,
        detection: GateObservation,
        timestamp_s: float,
    ) -> GateEstimate:
        apparent_size_px = None
        if detection.distance_est > 1e-6:
            apparent_size_px = self.detector.gate_w * self.camera_params.fx / detection.distance_est

        mode = TrackMode.DETECTED
        if detection.distance_est <= self.config.commit_distance_m:
            mode = TrackMode.COMMIT

        return GateEstimate(
            bearing_h_rad=float(detection.bearing_h),
            bearing_v_rad=float(detection.bearing_v),
            distance_m=float(detection.distance_est),
            confidence=float(np.clip(detection.confidence, 0.0, 1.0)),
            pixel_center=detection.pixel_centre,
            apparent_size_px=(
                float(apparent_size_px)
                if apparent_size_px is not None
                else None
            ),
            age_s=0.0,
            mode=mode,
        )

    def _fallback(self, timestamp_s: float) -> GateEstimate:
        if self._last is None or self._last_timestamp_s is None:
            return GateEstimate(mode=TrackMode.SEARCH)

        age_s = max(0.0, float(timestamp_s - self._last_timestamp_s))
        if age_s > self.config.max_track_age_s:
            return GateEstimate(age_s=age_s, mode=TrackMode.SEARCH)

        fade = max(0.0, 1.0 - age_s / max(self.config.max_track_age_s, 1e-6))
        confidence = self._last.confidence * self.config.tracked_confidence_decay * fade

        return GateEstimate(
            bearing_h_rad=self._last.bearing_h_rad,
            bearing_v_rad=self._last.bearing_v_rad,
            distance_m=self._last.distance_m,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            pixel_center=self._last.pixel_center,
            apparent_size_px=self._last.apparent_size_px,
            age_s=age_s,
            mode=TrackMode.TRACKED,
        )

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
    max_detector_candidates: int = 4
    sequence_memory_enabled: bool = True
    visual_pass_distance_m: float = 1.45
    visual_pass_distance_jump_m: float = 1.0
    post_pass_ignore_s: float = 0.70
    post_pass_min_distance_m: float = 2.4
    edge_margin_fraction: float = 0.12


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
        self._sequence_index = 0
        self._last_pass_timestamp_s: Optional[float] = None
        self._post_pass_until_s: Optional[float] = None

    def reset(self) -> None:
        self._last = None
        self._last_timestamp_s = None
        self._sequence_index = 0
        self._last_pass_timestamp_s = None
        self._post_pass_until_s = None

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
                detection = self._select_detection(detections, timestamp_s)
                if detection is not None:
                    estimate = self._estimate_from_detection(detection, timestamp_s)
                    self._last = estimate
                    self._last_timestamp_s = timestamp_s
                    return estimate
            else:
                self._register_pass_on_loss(timestamp_s)

        return self._fallback(timestamp_s)

    def _detect(self, frame: np.ndarray, frame_format: str) -> list[GateObservation]:
        fmt = frame_format.strip().lower()
        if fmt == "rgb":
            frame_bgr = frame[:, :, ::-1]
        elif fmt == "bgr":
            frame_bgr = frame
        else:
            raise ValueError("frame_format must be 'bgr' or 'rgb'.")

        detections = self.detector.detect(
            frame_bgr,
            cam=self.camera_params,
            max_gates=self.config.max_detector_candidates,
        )
        return [
            det
            for det in detections
            if det.confidence >= self.config.min_detect_confidence
        ]

    def _select_detection(
        self,
        detections: list[GateObservation],
        timestamp_s: float,
    ) -> Optional[GateObservation]:
        candidates = list(detections)

        if self._in_post_pass_window(timestamp_s):
            fresh_candidates = [
                det
                for det in candidates
                if det.distance_est >= self.config.post_pass_min_distance_m
            ]
            if not fresh_candidates:
                return None
            candidates = fresh_candidates

        next_gate = self._next_gate_after_visual_pass(candidates, timestamp_s)
        if next_gate is not None:
            return next_gate

        return min(candidates, key=lambda det: det.distance_est)

    def _next_gate_after_visual_pass(
        self,
        candidates: list[GateObservation],
        timestamp_s: float,
    ) -> Optional[GateObservation]:
        if not self.config.sequence_memory_enabled:
            return None
        if self._last is None or self._last.mode != TrackMode.COMMIT or not self._last.has_range:
            return None
        if self._last.distance_m is None:
            return None

        fresh_candidates = [
            det
            for det in candidates
            if det.distance_est >= self.config.post_pass_min_distance_m
        ]
        if not fresh_candidates:
            return None

        closest = min(candidates, key=lambda det: det.distance_est)
        distance_jumped = (
            closest.distance_est
            >= self._last.distance_m + self.config.visual_pass_distance_jump_m
        )
        close_edge_candidate = any(
            det.distance_est < self.config.post_pass_min_distance_m
            and self._near_frame_edge(det)
            for det in candidates
        )
        very_near_commit = self._last.distance_m <= self.config.visual_pass_distance_m * 0.90
        pass_ready = self._last.distance_m <= self.config.visual_pass_distance_m

        if pass_ready and (distance_jumped or close_edge_candidate or very_near_commit):
            self._register_visual_pass(timestamp_s, clear_last=False)
            return self._best_fresh_candidate(fresh_candidates)

        return None

    def _register_pass_on_loss(self, timestamp_s: float) -> None:
        if not self.config.sequence_memory_enabled:
            return
        if self._last is None or self._last.mode != TrackMode.COMMIT or not self._last.has_range:
            return
        if self._last.distance_m is None:
            return
        if self._last.distance_m > self.config.visual_pass_distance_m:
            return
        self._register_visual_pass(timestamp_s, clear_last=True)

    def _register_visual_pass(self, timestamp_s: float, *, clear_last: bool) -> None:
        if self._last_pass_timestamp_s is not None:
            elapsed = timestamp_s - self._last_pass_timestamp_s
            if 0.0 <= elapsed < self.config.post_pass_ignore_s:
                return

        self._sequence_index += 1
        self._last_pass_timestamp_s = timestamp_s
        self._post_pass_until_s = timestamp_s + self.config.post_pass_ignore_s
        if clear_last:
            self._last = None
            self._last_timestamp_s = None

    def _in_post_pass_window(self, timestamp_s: float) -> bool:
        return (
            self._post_pass_until_s is not None
            and timestamp_s < self._post_pass_until_s
        )

    def _best_fresh_candidate(
        self,
        candidates: list[GateObservation],
    ) -> GateObservation:
        return min(
            candidates,
            key=lambda det: (
                self._image_center_distance(det),
                det.distance_est,
                -det.confidence,
            ),
        )

    def _image_center_distance(self, detection: GateObservation) -> float:
        x, y = detection.pixel_centre
        dx = (float(x) - self.camera_params.cx) / max(self.camera_params.width, 1)
        dy = (float(y) - self.camera_params.cy) / max(self.camera_params.height, 1)
        return float(np.hypot(dx, dy))

    def _near_frame_edge(self, detection: GateObservation) -> bool:
        x, y = detection.pixel_centre
        margin = self.config.edge_margin_fraction * min(
            self.camera_params.width,
            self.camera_params.height,
        )
        return (
            x <= margin
            or x >= self.camera_params.width - margin
            or y <= margin
            or y >= self.camera_params.height - margin
        )

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
            sequence_index=self._sequence_index,
            age_s=0.0,
            mode=mode,
        )

    def _fallback(self, timestamp_s: float) -> GateEstimate:
        if self._last is None or self._last_timestamp_s is None:
            return GateEstimate(
                sequence_index=self._sequence_index,
                mode=TrackMode.SEARCH,
            )

        age_s = max(0.0, float(timestamp_s - self._last_timestamp_s))
        if age_s > self.config.max_track_age_s:
            return GateEstimate(
                sequence_index=self._sequence_index,
                age_s=age_s,
                mode=TrackMode.SEARCH,
            )

        fade = max(0.0, 1.0 - age_s / max(self.config.max_track_age_s, 1e-6))
        confidence = self._last.confidence * self.config.tracked_confidence_decay * fade

        return GateEstimate(
            bearing_h_rad=self._last.bearing_h_rad,
            bearing_v_rad=self._last.bearing_v_rad,
            distance_m=self._last.distance_m,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            pixel_center=self._last.pixel_center,
            apparent_size_px=self._last.apparent_size_px,
            sequence_index=self._sequence_index,
            age_s=age_s,
            mode=TrackMode.TRACKED,
        )

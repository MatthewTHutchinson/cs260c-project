"""Vision-based gate detector scaffolding for the AI Grand Prix project.

This detector is a lightweight HSV-segmentation baseline inspired by older
Virtual Qualifier messaging that suggested visually highlighted gates.
The latest technical spec does not guarantee that presentation, so this
detector should be treated as a provisional bridge rather than a
competition-ready perception stack.

Usage
-----
    detector = GateDetector(gate_physical_width=1.5)  # metres
    gates = detector.detect(bgr_frame, camera_params)
    # gates: list of GateObservation(bearing_h, bearing_v, distance_est, confidence)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@dataclass
class GateObservation:
    """Single detected gate in camera / body frame."""
    bearing_h: float      # horizontal bearing from camera centre (radians, +right)
    bearing_v: float      # vertical bearing (radians, +up)
    distance_est: float   # estimated distance (metres)
    confidence: float     # [0, 1]
    pixel_centre: tuple   # (cx, cy) in image coords


@dataclass
class CameraParams:
    """Pinhole camera intrinsics."""
    fx: float = 320.0     # focal length in pixels (x)
    fy: float = 320.0     # focal length in pixels (y)
    cx: float = 320.0     # principal point x
    cy: float = 180.0     # principal point y
    width: int = 640
    height: int = 360


# Default colour range for a high-saturation gate highlight.
# This remains a historical/provisional assumption and should be retuned
# against the real qualifier visuals.
_VQ1_HSV_LO = np.array([10,  150, 150], dtype=np.uint8)   # H, S, V lower bound
_VQ1_HSV_HI = np.array([35,  255, 255], dtype=np.uint8)   # H, S, V upper bound


class GateDetector:
    """Detect racing gates from a camera frame.

    Parameters
    ----------
    gate_physical_width : float
        Known gate width in metres (used for distance estimation via PnP).
    hsv_lo, hsv_hi : array-like
        HSV colour range for gate highlighting.  Adjust after observing VQ1.
    min_contour_area : int
        Minimum pixel area to consider a blob as a gate candidate.
    """

    def __init__(
        self,
        gate_physical_width: float = 1.5,
        hsv_lo: np.ndarray = _VQ1_HSV_LO,
        hsv_hi: np.ndarray = _VQ1_HSV_HI,
        min_contour_area: int = 200,
    ):
        if not _CV2_AVAILABLE:
            raise ImportError("opencv-python is required for GateDetector. "
                              "Install with: pip install opencv-python")
        self.gate_w = gate_physical_width
        self.hsv_lo = hsv_lo
        self.hsv_hi = hsv_hi
        self.min_area = min_contour_area

    def detect(
        self,
        bgr_frame: np.ndarray,
        cam: CameraParams = None,
        max_gates: int = 2,
    ) -> list[GateObservation]:
        """Detect up to `max_gates` gates in `bgr_frame`.

        Returns a list of GateObservation sorted by distance (nearest first).
        Returns an empty list if nothing is found.
        """
        cam = cam or CameraParams()

        # --- colour segmentation ---
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)

        # morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- contour extraction ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [c for c in contours if cv2.contourArea(c) > self.min_area]

        if not candidates:
            return []

        # Sort largest first (largest apparent area = closest gate)
        candidates.sort(key=cv2.contourArea, reverse=True)
        candidates = candidates[:max_gates]

        results = []
        for cnt in candidates:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_px, h_px), _ = rect
            apparent_size = max(w_px, h_px)

            if apparent_size < 2:
                continue

            # Bearing angles
            bh = np.arctan2(cx - cam.cx, cam.fx)   # +right
            bv = np.arctan2(cam.cy - cy, cam.fy)   # +up

            # Distance estimate from apparent size
            dist = (self.gate_w * cam.fx) / (apparent_size + 1e-6)

            area_ratio = cv2.contourArea(cnt) / (w_px * h_px + 1e-6)
            confidence = float(np.clip(area_ratio, 0.0, 1.0))

            results.append(GateObservation(
                bearing_h=float(bh),
                bearing_v=float(bv),
                distance_est=float(dist),
                confidence=confidence,
                pixel_centre=(float(cx), float(cy)),
            ))

        return sorted(results, key=lambda g: g.distance_est)

    def gate_obs_to_body_frame(
        self,
        gate_obs: GateObservation,
        cam_to_body_R: np.ndarray = None,
    ) -> np.ndarray:
        """Convert a GateObservation to a body-frame relative position vector.

        Parameters
        ----------
        gate_obs : GateObservation
        cam_to_body_R : (3,3) rotation matrix from camera frame to body frame.
            Defaults to identity (camera aligned with body x-axis).

        Returns
        -------
        rel_body : ndarray, shape (3,)
            Relative gate centre position in body frame (metres).
        """
        d = gate_obs.distance_est
        bh = gate_obs.bearing_h
        bv = gate_obs.bearing_v

        # Gate position in camera frame (x-forward, y-left, z-up convention)
        x_cam = d * np.cos(bv) * np.cos(bh)
        y_cam = -d * np.cos(bv) * np.sin(bh)
        z_cam = d * np.sin(bv)
        pos_cam = np.array([x_cam, y_cam, z_cam])

        if cam_to_body_R is None:
            return pos_cam
        return cam_to_body_R @ pos_cam

"""Camera profile definitions for detector and controller experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

from algorithm.gate_detector import CameraParams


@dataclass(frozen=True)
class CameraProfile:
    """Camera model used by the autonomy-side image geometry."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    tilt_up_deg: float
    lens_model: str
    render_effect: str = "normal"

    @property
    def fov_vert_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(self.cy / self.fy))

    @property
    def fov_horiz_deg(self) -> float:
        return 2.0 * math.degrees(math.atan(self.cx / self.fx))

    def camera_params(self) -> CameraParams:
        return CameraParams(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
        )


def _profile_from_vertical_fov(
    *,
    name: str,
    width: int,
    height: int,
    vertical_fov_deg: float,
    tilt_up_deg: float,
    lens_model: str,
    render_effect: str,
) -> CameraProfile:
    cx = width / 2.0
    cy = height / 2.0
    focal_px = cy / math.tan(math.radians(vertical_fov_deg) / 2.0)
    return CameraProfile(
        name=name,
        width=width,
        height=height,
        fx=focal_px,
        fy=focal_px,
        cx=cx,
        cy=cy,
        tilt_up_deg=tilt_up_deg,
        lens_model=lens_model,
        render_effect=render_effect,
    )


VQ1_PINHOLE = CameraProfile(
    name="vq1_pinhole",
    width=640,
    height=360,
    fx=320.0,
    fy=320.0,
    cx=320.0,
    cy=180.0,
    tilt_up_deg=20.0,
    lens_model="pinhole",
    render_effect="normal",
)

GATENET_FISHEYE = _profile_from_vertical_fov(
    name="gatenet_fisheye",
    width=640,
    height=360,
    vertical_fov_deg=120.0,
    tilt_up_deg=20.0,
    lens_model="fisheye_wide_fov_experiment",
    render_effect="fisheye",
)

CAMERA_PROFILES = {
    profile.name: profile
    for profile in (
        VQ1_PINHOLE,
        GATENET_FISHEYE,
    )
}


def get_camera_profile(name: str | None = None) -> CameraProfile:
    """Return a named camera profile.

    If `name` is omitted, `ELODIN_CAMERA_PROFILE` or `CS260C_CAMERA_PROFILE`
    selects the profile. The VQ1 pinhole model remains the default.
    """
    selected = (
        name
        or os.environ.get("ELODIN_CAMERA_PROFILE")
        or os.environ.get("CS260C_CAMERA_PROFILE")
        or VQ1_PINHOLE.name
    )
    key = selected.strip().lower()
    if key not in CAMERA_PROFILES:
        available = ", ".join(sorted(CAMERA_PROFILES))
        raise ValueError(f"unknown camera profile '{selected}', available: {available}")
    return CAMERA_PROFILES[key]

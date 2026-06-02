# Sign Convention Audit

Date: 2026-06-02

## Why This Exists

The Elodin editor run exposed a classic sign-convention failure: the camera was
documented as `20` degrees upward, but the first gate appeared at the very top
of the FPV image even when simple course geometry predicted it should appear
below center.

This matters because camera pitch, image bearings, RC pitch, and climb commands
all interact. A single wrong sign can make the autonomy stack look like a weak
detector or bad planner when the real problem is coordinate interpretation.

## Camera Pitch

Spec-facing convention:

- camera tilt is `20` degrees upward relative to the body frame
- image vertical bearing is positive when the gate center is above image center
- pixel `y` increases downward

Evidence from the Elodin run before the fix:

- first gate center: approximately `(x=10m, z=1.8m)`
- drone height during first detection: roughly `z=0.3-0.5m`
- body-frame elevation to gate: about `7-10` degrees upward
- if the camera were pitched upward by `20` degrees, the gate should appear
  below image center, around `pixel_y=240-250`
- observed first detection was near the top: `pixel_y=29.5`

Conclusion:

The Elodin `sensor_camera.rot_offset` pitch sign was opposite our assumption.
The harness now represents AGP's upward tilt as:

```python
ELODIN_ROT_OFFSET_PITCH_DEG = -CAM_TILT_UP_DEG
rot_offset=[0.0, ELODIN_ROT_OFFSET_PITCH_DEG, 0.0]
```

This fix lives in the sibling Elodin harness and is captured in
`patches/elodin-ai-grand-prix-cs260c.patch`.

## Image Bearings

Detector convention in `algorithm/gate_detector.py`:

- horizontal bearing: `atan2(pixel_x - cx, fx)`, positive means gate appears
  right of image center
- vertical bearing: `atan2(cy - pixel_y, fy)`, positive means gate appears
  above image center

This convention is internally consistent with OpenCV image coordinates because
pixel `y` grows downward.

## Throttle / Vertical Control

Controller convention in `algorithm/reactive_controller.py`:

- positive `bearing_v_rad` means the gate is above image center
- vertical thrust is based on body-frame elevation, not raw image bearing
- body-frame elevation is approximated as
  `gate.bearing_v_rad + camera_tilt_up_rad`
- large body-vertical error suppresses forward pitch so
  the drone climbs to keep the gate in view before accelerating through it
- the Elodin detector estimates range from the visible outer gate frame
  (`2.7m`), not the inner flyable opening (`1.5m`)
- Betaflight RC throttle uses the full normalized range
  `1000 + thrust_norm * 1000`

This distinction matters at takeoff. With a `20` degree upward camera, the first
gate can appear below the optical center while still being physically above the
drone. Raw image bearing alone would reduce thrust in that case; tilt-compensated
body elevation correctly commands a climb.

Audit result:

- a high gate produces higher thrust than a body-aligned gate
- a below-optical-center first gate still commands climb when body elevation is
  positive after camera-tilt compensation
- a high gate produces less aggressive forward pitch than a body-aligned gate
- live Elodin editor validation passed gate 0 after these fixes:
  `t=6.78s`, position `(10.00, -0.11, 1.44)`

## Internal Pitch vs Betaflight RC Pitch

Internal command convention:

- negative `pitch_rate_rad_s` means nose-down / accelerate forward

Betaflight RC mapping in this harness:

- RC pitch above center commands the forward / nose-down direction
- therefore internal negative pitch must map to RC pitch above `1500`

Current adapter:

```python
pitch = 1500.0 - command.pitch_rate_rad_s * pitch_scale
```

Audit result:

- internal `pitch=-0.25` maps to `rc_pitch > 1500`
- internal `pitch=+0.25` maps to `rc_pitch < 1500`

## Remaining Unknowns

Yaw and roll are not currently treated as sign-flipped. A temporary Betaflight
RC roll/yaw inversion was tested and rejected because it amplified lateral
error badly (`y=-2.98m` by `t=5s` in the short editor run). The remaining
horizontal problem appears to be lateral control/target-selection quality, not a
confirmed wire-sign inversion.

The next controlled horizontal audit should use an intentionally offset gate or
synthetic course:

- gate appears right of center -> yaw/roll command should rotate/translate
  toward the right-side bearing rather than away from it
- gate appears left of center -> command signs should mirror
- after passing gate 0, the tracker should reacquire gate 1 rather than
  overreact to the nearby gate frame or stale target

## Repeatable Audit Command

Run:

```bash
scripts/audit_sign_conventions.py
```

The script checks the static sign assumptions and compares the latest editor
trace, if present, against the expected camera-up vs camera-down pixel geometry.

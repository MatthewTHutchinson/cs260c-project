# Competition Notes

This file is the organized reference for external AI Grand Prix / DCL-facing constraints.

## Confidence legend

- `Confirmed`: stated in the latest technical spec PDF currently in the repo.
- `Historical`: stated in older Gmail PDFs now stored in `docs/`, but not necessarily repeated in the latest spec.
- `Inference`: derived from the repo or from comparing sources, but not explicitly confirmed by the latest spec.

## Source hierarchy

1. `Confirmed`:
   `docs/reference/260508_Technical_Spec_0002.pdf`
   Document ID `VADR-TS-002`, Issue `00.02`, dated `2026-05-08`
2. `Historical`:
   the Gmail PDF exports in `docs/`
3. `Inference`:
   repo notes, design assumptions, and implementation gaps called out explicitly below

## Confirmed from `VADR-TS-002` Issue `00.02`

### Simulator and course

- `Confirmed`: physics update frequency is `120 Hz`.
- `Confirmed`: the environment includes a start gate, sequential race gates, a finish gate, vertical and horizontal obstacles, boundary elements, terrain, and environmental structures.
- `Confirmed`: the visual environment includes a forward-facing first-person camera, gates, course guidance structures, static scene objects, and dynamic lighting.
- `Confirmed`: course geometry, physics parameters, and environmental conditions are deterministic and identical for all participants.
- `Confirmed`: the simulator uses a local Cartesian coordinate system internally.
- `Confirmed`: GPS simulation is not available.
- `Confirmed`: absolute global position is not exposed.

### Drone and gate geometry

- `Confirmed`: drone chassis dimensions are `280 mm x 280 mm x 160 mm`.
- `Confirmed`: outer gate dimensions are `2700 mm x 2700 mm x 260 mm`.
- `Confirmed`: inner square gate opening is `1500 mm x 1500 mm x 260 mm`.

### Frames and camera model

- `Confirmed`: MAVLink frame convention is NED.
- `Confirmed`: `MAV_FRAME_BODY_NED` uses `X` forward, `Y` right, `Z` down.
- `Confirmed`: camera and body share the same origin.
- `Confirmed`: camera is tilted `20` degrees upward relative to the body frame.
- `Confirmed`: camera model is standard pinhole with no lens distortion.
- `Confirmed`: image resolution is `640 x 360`.
- `Confirmed`: principal point is `[320, 180]`.
- `Confirmed`: focal lengths are `[320, 320]`.
- `Confirmed`: vertical field of view is `90` degrees.
- `Open`: the published intrinsics and published `VFoV` are not perfectly self-consistent under a standard pinhole model.
  The repo now supports explicit intrinsics overrides so we can mirror the stated `[fx, fy, cx, cy]`
  directly instead of deriving everything from one FOV formula.

### Communications and control

- `Confirmed`: simulator communication uses MAVLink 2 through MAVSDK-compatible interfaces.
- `Confirmed`: supported transport is UDP.
- `Confirmed`: supported message list includes:
  `HEARTBEAT`,
  `ATTITUDE`,
  `HIGHRES_IMU`,
  `SET_POSITION_TARGET_LOCAL_NED`,
  `SET_ATTITUDE_TARGET`,
  and `TIMESYNC`.
- `Inference`: the duplicated `HIGHRES_IMU` row in the extracted PDF appears to be a formatting issue rather than a second distinct message.
- `Confirmed`: command rate is `<100 Hz`.
- `Confirmed`: minimum heartbeat rate is `2 Hz`.
- `Confirmed`: telemetry includes vehicle attitude, orientation, linear velocities, and system status flags.

### Vision stream

- `Confirmed`: camera stream frequency is `30 Hz`.
- `Confirmed`: camera stream resolution is `640 x 360`.
- `Confirmed`: vision transport uses UDP on default port `5600`.
- `Confirmed`: each packet contains a `24`-byte little-endian header followed by JPEG payload bytes.
- `Confirmed`: the header contains `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, and `sim_time_ns`.

### Runtime and qualification constraints

- `Confirmed`: a Python-based runtime environment may be assumed.
- `Confirmed`: Python `3.14.2` is known to work.
- `Confirmed`: participants may choose other environments.
- `Confirmed`: the DCL simulator runs on Windows 11.
- `Confirmed`: Linux is currently not supported.
- `Confirmed`: client responsibilities include:
  establishing MAVLink communication,
  maintaining heartbeats,
  sending control commands,
  processing telemetry,
  and processing the vision stream.
- `Confirmed`: the intended conceptual stack is:
  `Vision + Telemetry -> Perception -> Planning -> Control -> Pilot Commands -> Stabilized Controller`.
- `Confirmed`: Round One objective is to navigate the racecourse successfully.
- `Confirmed`: Round One course uses a start gate, intermediate gates, and a finish gate.
- `Confirmed`: maximum run duration is `8` minutes.

## Historical notes from older Gmail PDFs

These are still useful for planning context, but should be treated as provisional unless they are also stated in the latest spec.

### Likely still useful

- `Historical`: the competition framing emphasizes gate recognition, control, and path planning together.
- `Historical`: evaluation is meant to be autonomous-only, with no manual intervention.
- `Historical`: the Windows-based downloadable DCL application and controlled evaluation environment were communicated before the full technical spec, and that remains broadly consistent with the latest spec.

### Potentially stale or superseded

- `Historical`: VQ1 was described as lower-complexity, desaturated, and possibly using highlighted gates or active guidance aids.
- `Historical`: VQ2 was described as more visually complex, with distractions and no guidance aids.
- `Historical`: older emails said the API would not provide depth, engine RPM, or battery state of charge.
- `Historical`: one email described the control side as classical drone commands such as throttle, roll, pitch, and yaw.
- `Historical`: older public website FAQ copy also included additional claims such as likely motor RPM readouts, a `12MP` wide-angle camera, no wind in virtual qualifiers, and limited starting-coordinate access.
- `Inference`: because some of that public FAQ copy conflicted with the latest spec, it should not be treated as a current interface contract.
- `Inference`: the latest spec's explicit MAVLink message list suggests that older control wording was preliminary and may not be the right abstraction to design around now.

## Repo implications

- `Inference`: the current PyBullet work is still a simulator-side research stack, not yet a true drop-in DCL runtime client.
- `Inference`: the official camera model now in the repo docs should bias future simulator camera settings toward:
  `640 x 360`,
  `30 Hz`,
  pinhole intrinsics,
  and a `20`-degree upward tilt.
- `Inference`: the repo now supports explicit camera intrinsics overrides and separate policy-image resizing,
  so competition-aligned evaluation can render spec-sized frames while still feeding existing `128 x 96` multimodal policies.
- `Inference`: the current multimodal and perception-bridge work is directionally aligned with the official spec because the spec clearly expects combined telemetry and forward-camera processing.
- `Inference`: the current best completed policies still rely on simulator-side structure and are not yet validated through a real MAVLink / UDP SITL bridge.

## Open questions to keep flagged

- `Open`: which of the supported control messages will be the practical primary interface for teams:
  `SET_POSITION_TARGET_LOCAL_NED`,
  `SET_ATTITUDE_TARGET`,
  or both?
- `Open`: whether the older "no depth" statement remains binding, since it does not appear explicitly in `Issue 00.02`.
- `Open`: how much of the older VQ1 visual-guidance description is still accurate for the current qualifier.
- `Open`: exact packaging, submission, and process-management details for the final DCL client runtime.
- `Open`: whether the current Windows-only note will remain true through later competition phases.

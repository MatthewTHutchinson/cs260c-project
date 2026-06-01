# Competition Context

Date: 2026-05-22

## Source Hierarchy

Use sources in this order:

1. Current formal spec: `docs/reference/260508_Technical_Spec_0002.pdf`
2. April 10, 2026 pre-spec briefing notes from the team
3. Older Gmail PDF exports in `docs/reference/`
4. Public website / public FAQ, only for broad context
5. Repo assumptions and implementation notes

When sources conflict, the formal May 8 spec wins for interface and camera details. The April 10 notes remain important for competition structure, expected VQ1/VQ2 difficulty, and the practical emphasis of each round.

## April 10 Pre-Spec Briefing

The April 10 notes said simulator access credentials and setup instructions would be shared shortly before Virtual Qualifier 1. Expected materials:

- download access to the simulation environment
- installation and configuration guidelines
- instructions for connecting to the simulator

Timeline from the April 10 notes:

- Virtual Qualifier 1: launch in May, open until the end of Virtual Qualifier 2
- Virtual Qualifier 2: launch in June, open until mid to late July
- Physical Qualifier: September, California, USA
- Grand Prix Final: November, Ohio, USA

Competition structure from the April 10 notes:

- VQ1: short, simplified simulator course; focus on successful completion
- VQ2: longer and more complex simulator course; fastest valid time counts
- Physical Qualifier: real drones in a controlled environment with no audience
- Grand Prix Final: real drones under race conditions with environmental distractions and audience
- technical specifications currently cover only the virtual qualifier and will expand later

Environment and course expectations from the April 10 notes:

- same course for all teams within each qualifier
- VQ1 has minimal distractions and clear gate visibility
- VQ2 adds lighting changes, 3D objects, and obstacles
- gates are visually distinguishable and consistent within each round
- gates change positions between VQ1 and VQ2
- start and finish gates may differ slightly
- non-gated objects may appear in the environment
- VQ1 has fewer than 10 gates
- VQ2 has fewer than 20 gates
- full 3D environment with elevation changes

Simulator and environment insights from the April 10 notes:

- simulator is provided as a downloadable package and runs locally
- active internet connection is required for anti-cheat measures
- Windows is supported
- Linux was not working at the time of the briefing
- teams can run multiple instances in parallel
- gates remain consistent within a qualifier
- gate positions change between VQ1 and VQ2
- start and finish gate may differ slightly
- non-gated objects may be present

Rules and performance framing from the April 10 notes:

- unlimited attempts within the qualification window
- VQ1 prioritizes completion
- VQ2 uses fastest valid time
- maximum run time is 8 minutes per attempt
- gates must be passed in correct order
- no human interaction during runs
- unfair manipulation or advantage can cause disqualification

April 10 data/control framing:

- inputs: FPV visual stream plus telemetry
- outputs: standard drone control, described as throttle, roll, pitch, yaw
- no GPS or absolute positioning
- no depth data or sensor shortcuts
- camera details for the physical stage were expected in later updates

Important reconciliation: the May 8 spec is more specific about the control transport. It lists MAVLink support, including `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`. Treat "throttle, roll, pitch, yaw" as an older conceptual framing of drone control, not necessarily the final wire-level API.

Software, fair-play, and eligibility notes from April 10:

- any software stack, tools, or frameworks can be used
- code must be accessible for review if required
- no human interaction is allowed during runs
- manipulation or unfair advantage can lead to disqualification
- internships or contractor roles with partners are allowed
- full-time employees of founding partners are not eligible
- each participant can only be part of one team

## Current May 8 Spec Anchors

The May 8 spec confirms:

- simulator physics update frequency: `120 Hz`
- camera stream frequency: `30 Hz`
- camera resolution: `640 x 360`
- camera/body origin alignment
- camera tilt: `20` degrees upward relative to body frame
- pinhole model with no lens distortion
- stated intrinsics: `[fx, fy, cx, cy] = [320, 320, 320, 180]`
- no GPS simulation
- absolute global position is not exposed
- telemetry includes attitude, orientation, linear velocities, and system status flags
- MAVLink 2 through MAVSDK-compatible interfaces over UDP
- supported messages include `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `SET_POSITION_TARGET_LOCAL_NED`, `SET_ATTITUDE_TARGET`, and `TIMESYNC`
- command rate is below `100 Hz`
- minimum heartbeat rate is `2 Hz`
- Round One objective is successful racecourse navigation
- Round One includes a start gate, intermediate gates, and a finish gate
- maximum run duration is `8` minutes

The stated intrinsics and vertical field of view are not perfectly self-consistent under a standard pinhole model. Prefer explicit intrinsics when mirroring the spec.

MAVSDK/MAVLink interpretation:

- The spec explicitly says the client initializes MAVSDK.
- We should not hand-build raw MAVLink byte packets for the first implementation.
- MAVLink `common.xml` remains the schema reference for what messages such as `SET_ATTITUDE_TARGET` and `SET_POSITION_TARGET_LOCAL_NED` contain.
- `SET_ATTITUDE_TARGET` carries attitude/body-rate plus thrust style control and is the closest current match to "Pilot Commands -> Stabilized Controller."
- `SET_POSITION_TARGET_LOCAL_NED` can represent local/body position, velocity, acceleration, yaw, and yaw-rate setpoints, but global coordinate navigation remains out because GPS/global position is not exposed.

IMU interpretation:

- Yes, the current spec lists `HIGHRES_IMU` as a `Simulator -> Client` message.
- In MAVLink `common.xml`, `HIGHRES_IMU` includes acceleration fields `xacc`, `yacc`, `zacc` in `m/s/s` and angular-rate fields `xgyro`, `ygyro`, `zgyro` in `rad/s`, along with magnetometer, pressure, temperature, timestamp, and update flags.
- The spec table labels one `HIGHRES_IMU` row as vehicle status and another as measurements, likely because of PDF/table formatting. Treat the message as available telemetry, but verify exact field updates and rate in the VQ1 logger.

## Strategic Implications

VQ1 should be treated as a completion problem first. A slow but reliable system that detects gates, passes them in order, and avoids disqualification is more valuable than a fast system that only works in local simulator assumptions.

The engineering stack should be organized around:

```text
FPV camera + telemetry
  -> gate recognition
  -> local navigation state
  -> path planning
  -> drone control adapter
  -> MAVLink commands
```

No design should depend on GPS, global position, depth, or simulator-only gate labels.

## Open Questions

- What exact commands does the VQ1 simulator respond to best in practice?
- Does `SET_POSITION_TARGET_LOCAL_NED` accept useful velocity/body-frame masks without exposed absolute position?
- How visually distinct are VQ1 gates in real captured frames?
- Are start and finish gates visibly different, and do they require separate recognition logic?
- Are non-gate objects visually similar enough to cause false positives?
- Can repeated attempts be used to build an internal course map without violating the no-human-interaction rule during runs?

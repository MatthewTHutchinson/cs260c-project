# Drone Control

Date: 2026-05-22

## Goal

Build a control layer that can safely pass VQ1 gates using telemetry and FPV-derived navigation commands. Completion comes first; speed comes later.

The control layer should convert navigation intent into simulator commands while hiding interface differences from the planner.

## Source Reconciliation

April 10 notes described outputs as standard drone control:

```text
throttle, roll, pitch, yaw
```

The May 8 spec is more specific about the transport and supported MAVLink messages:

```text
SET_POSITION_TARGET_LOCAL_NED
SET_ATTITUDE_TARGET
```

References:

- MAVLink 2 overview: `https://mavlink.io/en/guide/mavlink_2.html`
- MAVLink common message set: `https://mavlink.io/en/messages/common.html`
- MAVSDK Offboard API: `https://mavsdk.mavlink.io/main/en/cpp/api_reference/classmavsdk_1_1_offboard.html`

Therefore the control architecture should not hard-code one interpretation too early. It should expose an internal command interface and implement adapters:

```text
navigation intent
  -> internal desired motion
  -> control adapter
  -> MAVLink message
```

After reviewing MAVLink `common.xml` and MAVSDK docs, the practical conclusion is:

- We should use MAVSDK as the primary client layer instead of manually constructing raw MAVLink packets.
- `common.xml` is still useful for understanding what MAVSDK sends and what fields/sign conventions matter.
- The most competition-aligned first adapter is attitude/rate control plus thrust, because it maps cleanly to the spec's conceptual "Pilot Commands -> Stabilized Controller" layer.
- Body-frame velocity plus yaw-rate remains a strong fallback because it is easy to drive from image-space gate tracking and does not require GPS/global coordinates.
- Global coordinate navigation is out. Local/body setpoints may still be usable if the simulator's MAVSDK/offboard bridge accepts them, but they must be probed in VQ1.

## Internal Command Interface

Use one internal representation for the planner:

```text
desired_forward_speed
desired_lateral_speed
desired_vertical_speed
desired_yaw_rate
confidence
mode
```

This is compatible with reactive gate following and does not require absolute position.

Modes:

- `track_gate`: center and approach the current gate
- `pass_gate`: commit through the gate
- `search`: reacquire a gate
- `recover`: stabilize after detection loss or excessive attitude
- `stop`: safe hover/slowdown if supported

## Candidate MAVLink Mappings

### Option A: Offboard Attitude Rate + Thrust

Primary VQ1 control path to test.

Intent:

```text
roll_rate  = lateral correction / banking command
pitch_rate = forward/back correction
yaw_rate   = gate-centering heading correction
thrust     = altitude / climb command
```

Why:

- closest to the April 10 throttle/roll/pitch/yaw framing
- maps through MAVSDK Offboard attitude-rate APIs
- corresponds to the MAVLink `SET_ATTITUDE_TARGET` message family
- does not require position or GPS
- gives direct control over the stabilized controller input layer

Risk:

- thrust scaling can be simulator-specific
- sign conventions must be probed carefully
- rate commands can destabilize quickly if gains are too high
- requires a safety wrapper and conservative command limits

Initial policy:

```text
gate horizontal error -> yaw_rate
gate vertical error   -> thrust trim
gate size/confidence  -> pitch/forward aggressiveness
lateral image drift   -> roll_rate or yaw-first correction
```

### Option B: Offboard Attitude + Thrust

Second attitude-family option.

Intent:

```text
roll_angle  = lateral/banking command
pitch_angle = forward/back command
yaw_angle   = heading command
thrust      = altitude / climb command
```

Why:

- smoother than pure rate commands if the simulator's stabilized controller tracks angles cleanly
- still maps to the attitude-control family
- still avoids global position

Risk:

- yaw angle can be awkward without a persistent heading reference
- may be less reactive than rate control for visual servoing
- still needs careful thrust calibration

### Option C: Body Velocity + Yaw Rate

Strong fallback and possibly the easiest visual-servo control law.

Intent:

```text
forward_m_s = desired_forward_speed
right_m_s   = desired_lateral_speed
down_m_s    = desired_vertical_speed
yawspeed    = desired_yaw_rate
```

Why:

- MAVSDK exposes body-frame velocity plus yaw-speed setpoints
- image-space gate tracking maps naturally to forward/right/down/yaw-speed
- no global coordinates are required
- safer to tune than raw attitude/thrust in some simulators

Risk:

- this is less directly aligned with "Pilot Commands" wording
- underlying support may depend on the simulator's offboard bridge behavior
- exact MAVLink message and type mask behavior should be verified empirically

### Option D: Local Position Target

Useful only if local position setpoints are confirmed to work without hidden GPS/global assumptions.

Intent:

```text
target = current_estimated_local_position + short_horizon_delta
yaw = current_yaw + yaw_delta
```

Why:

- close to the old repo action
- stable for learning-oriented simulators

Risk:

- no absolute global position is exposed
- local position availability and estimator behavior must be confirmed
- can hide control issues behind a high-level autopilot behavior

This should not be the first competition-facing control path.

## VQ1 Control Philosophy

For VQ1, prioritize:

1. stable attitude
2. low speed near uncertainty
3. gate centering
4. correct order
5. finish completion

Not first priority:

- minimum-time racing line
- aggressive corner cutting
- high-speed split optimization
- direct transfer of old PPO actions

## Reactive Control Baseline

The first VQ1-capable baseline can be non-learning:

```text
image gate center error
  -> yaw/lateral correction
gate apparent size
  -> pitch/thrust or forward-speed schedule
vertical image error
  -> thrust or vertical correction
detection confidence
  -> speed cap and search behavior
```

This should be implemented before training new policies. It gives us a sanity baseline and a safe fallback.

## Safety Limits

Initial caps:

- low pitch/forward command until gate detection is stable
- bounded yaw rate
- bounded thrust trim
- conservative roll/pitch or roll/pitch-rate
- immediate slowdown if detection is lost for too long
- emergency stop/recover mode for extreme attitude or telemetry flags

All sign conventions must be tested with small commands:

- body X forward
- body Y right
- body Z down in `MAV_FRAME_BODY_NED`
- yaw positive direction
- vertical command sign

## Local Harness Control Reset

The old body-frame waypoint action is no longer the main research target. The active local harness is Elodin, with the PyBullet stack preserved under `legacy/pybullet/`.

New harness work should add a control path that better matches the internal command interface:

```text
[roll_or_roll_rate, pitch_or_pitch_rate, thrust, yaw_rate]
```

Then compare:

- old waypoint-delta PID
- attitude-rate/thrust controller
- attitude/thrust controller
- velocity/yaw-rate controller

## Immediate Tasks

1. Build a small MAVLink command probe once VQ1 opens.
2. Verify heartbeat and telemetry rates.
3. Test harmless yaw-rate, thrust-trim, roll, and pitch commands one at a time.
4. Record command response logs.
5. Compare attitude-rate/thrust against body-velocity/yaw-rate.
6. Choose the first real control adapter.
7. Mirror that adapter in Elodin.
8. Implement the reactive gate-centering baseline.
9. Only then consider learned control policies.

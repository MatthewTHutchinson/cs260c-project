# Autonomous Racing Algorithm

Date: 2026-05-29

This is the final-project-facing algorithm description. The project story is not "we reset the architecture." The project story is:

```text
FPV gate recognition
  -> temporal gate tracking
  -> reactive navigation state
  -> body-rate/thrust control
  -> simulator-specific command adapter
```

The reset matters only because it removed old assumptions that were not competition-facing.

## Core Claim

The VQ1 algorithm is a perception-guided body-rate autopilot for autonomous gate racing.

It does not depend on GPS, global position, simulator gate labels, depth, or a pre-known map. It consumes:

- FPV camera frames
- attitude/orientation telemetry
- angular rates and linear velocity when available
- `HIGHRES_IMU` acceleration/angular-rate signals when reliable

It outputs:

- roll rate
- pitch rate
- yaw rate
- normalized collective thrust

That output is the internal command boundary. Deployment maps it to MAVSDK `SET_ATTITUDE_TARGET` / attitude-rate style commands. The local Elodin harness maps the same command to Betaflight-style RC packet fields.

## Why This Algorithm

VQ1 is completion-first. A transparent gate-following autopilot is the fastest route to a valid run because it gives us:

- direct debug visibility from image error to command
- no dependency on absolute position
- no large training run before seeing real VQ1 frames
- a natural path to later learning by replacing the hand-tuned controller with a compact policy

The old IL/RL policies remain useful baselines, but they are not the current competition-facing algorithm.

## Runtime Loop

At each control step:

1. Receive latest FPV frame.
2. Receive latest telemetry.
3. Detect gate candidates in image space.
4. Update temporal tracker.
5. Choose navigation mode: `search`, `detected`, `tracked`, `commit`, or `recover`.
6. Convert gate bearing/range/confidence into body-rate/thrust command.
7. Apply safety limits.
8. Send command through the backend adapter.
9. Log raw frame, detection, telemetry, command, and simulator status.

## Gate Recognition

The first detector is deliberately classical:

- HSV/contrast thresholding after observing VQ1 frames
- contour extraction
- rectangle/quadrilateral scoring
- bearing from camera intrinsics
- approximate range from apparent gate width

The output is:

```text
bearing_h_rad
bearing_v_rad
distance_m, if estimated
confidence
pixel_center
apparent_size_px
age_s
mode
```

The current implementation begins in `algorithm/gate_tracker.py` and wraps `algorithm/gate_detector.py`.

## Temporal Tracking

The tracker keeps a short-lived gate estimate when a frame misses detection. This prevents one dropped frame from causing a full control-mode reset.

Tracking states:

- `detected`: current frame has a usable gate candidate.
- `tracked`: no current detection, but a recent estimate is still fresh enough.
- `search`: no reliable gate estimate.
- `commit`: the visible gate is close enough that the controller should keep passing through instead of dithering.
- `recover`: reserved for future safety handling after excessive attitude, collision flags, or sustained detection loss.

No competition-facing mode may fall back to simulator track truth.

## Control Law

The first controller is a conservative visual-servo law:

```text
yaw_rate   = k_yaw * horizontal_gate_bearing * confidence
roll_rate  = k_roll * horizontal_gate_bearing * confidence
thrust     = hover_thrust + k_vertical * vertical_gate_bearing
pitch_rate = forward approach command from gate range/size/confidence
```

The vertical terms now deliberately use two different cues:

- climb/thrust remains conservative and camera/body-relative, so the first gate
  still gets enough climb authority while it is low in the FPV image
- forward suppression uses attitude/orientation when available, so a gate
  centered in the upward-tilted camera while the drone is pitched down can still
  receive forward authority during post-gate reacquisition

If the gate is not usable, the drone enters search:

```text
roll_rate  = 0
pitch_rate = 0
yaw_rate   = slow scan rate
thrust     = hover estimate
```

All commands are clipped before leaving the algorithm.

The initial implementation is in:

```text
algorithm/reactive_controller.py
algorithm/autopilot.py
algorithm/control_adapter.py
```

## Elodin Caveats And Workarounds

The Elodin practice harness is useful, but it is not a drop-in competition simulator.

### Betaflight UDP Instead Of MAVLink

This is a real deployment caveat, but it is manageable.

It matters because the official spec is MAVLink 2 through MAVSDK-compatible UDP, while Elodin's solver talks to a Betaflight SITL bridge. Code written directly around Elodin packet structures will not deploy unchanged.

The workaround is the internal `RacingCommand` boundary:

```text
algorithm output: RacingCommand(roll_rate, pitch_rate, yaw_rate, thrust)
  -> Elodin adapter: Betaflight RC command
  -> official adapter: MAVSDK attitude-rate/thrust command
```

So Elodin can test perception, timing, and qualitative control behavior, but the official MAVSDK adapter still has to be written and probed in VQ1.

### ENU World State Instead Of NED

This is a classic sign-convention bug source, not a blocker.

The algorithm should not consume world pose at all. For logs and adapter checks, keep frame transforms centralized:

```text
algorithm/frames.py
```

The rule is: convert at the adapter boundary, then keep the algorithm in image/body-frame quantities.

### World Pose In Solver Update

This is the biggest caveat if we use it. The official spec says absolute global position is not exposed.

The workaround is simple and strict:

- do not pass world position into `AutonomousRacingPilot`
- use world pose only for offline scoring/debug plots
- mark any experiment that uses world pose as privileged and non-competition-facing

If we follow that boundary, Elodin's extra pose field stops being a contamination problem.

## Simulator Direction

Elodin is now the primary local harness while Windows/VQ1 access is unavailable.

PyBullet is preserved only as legacy work under:

```text
legacy/pybullet/
```

That old stack can still provide historical context, but it should not drive the final-project claims. The main harness work is now:

- use the stable no-FPV Elodin smoke path for Betaflight/control checks
- use `elodin editor` separately for FPV/render inspection
- keep `AutonomousRacingPilot` connected to Elodin `SensorUpdate` without world pose
- map `RacingCommand` to Elodin `RCCommand`
- verify detector behavior on Elodin FPV frames
- keep official VQ1/MAVSDK as the deployment target when Windows access arrives

## Four-Day Final Presentation Story

The final project should present the algorithm cleanly:

1. Problem: autonomous drone gate racing from FPV plus telemetry, no GPS/depth/global pose.
2. Perception: classical gate detector and temporal tracker.
3. Navigation: search/track/commit state machine.
4. Control: body-rate/thrust visual servoing.
5. Adaptation: same internal command mapped to Elodin now and MAVSDK in VQ1.
6. Evaluation: Elodin harness behavior, detection stability, command traces, and caveat analysis.

The architecture reset is background context, not the headline.

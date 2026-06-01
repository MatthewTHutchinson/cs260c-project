# Final Presentation Outline

Date: 2026-06-01

This presentation should be about the autonomous drone racing algorithm, not the project reset.

## Core Thesis

We built a simulator-independent FPV gate-racing autopilot that converts camera frames and allowed telemetry into body-rate/thrust commands for autonomous course completion.

```text
FPV frame
  -> gate detector
  -> temporal tracker
  -> navigation mode
  -> reactive body-rate/thrust controller
  -> backend command adapter
```

## Slide 1: Problem

Autonomous drone gate racing requires passing gates in the correct order using onboard observations.

Key constraints:

- no GPS
- no absolute global position
- no depth shortcut
- FPV camera at `640 x 360`, `30 Hz`
- telemetry includes attitude/orientation, angular rates, linear velocity, IMU, and status flags
- commands are pilot-style throttle/roll/pitch/yaw or attitude-rate/thrust through the simulator interface

## Slide 2: Design Decision

For VQ1, completion matters more than lap time.

The chosen baseline is a transparent visual-servoing autopilot rather than a large RL/world-model system.

Why:

- easier to debug from image error to command
- works before a large training run exists
- avoids dependence on simulator track truth
- creates clean logs for later imitation or RL fine-tuning

## Slide 3: Gate Recognition

Current detector:

- HSV segmentation
- contour extraction
- gate candidate scoring
- bearing from camera intrinsics
- range estimate from apparent gate size

Output:

```text
bearing_h_rad
bearing_v_rad
distance_m
confidence
pixel_center
apparent_size_px
```

Evidence to show:

- `assets/presentation/gate_sequence_detected.jpg`
- trace row with detection confidence, bearing, and range

## Slide 4: Temporal Tracking

The detector does not control the drone frame-by-frame by itself.

The tracker smooths short detection dropouts and emits a navigation-facing state:

- `search`: no usable gate
- `detected`: current frame has a gate
- `tracked`: recent gate estimate is still fresh
- `commit`: close enough to keep passing through
- `recover`: reserved safety mode

This is the layer that prevents one missed frame from causing chaotic control.

Evidence to show:

- `assets/presentation/gate_sequence_trace.png`
- `assets/presentation/gate_sequence_commit.jpg`
- `assets/presentation/gate_sequence_search.jpg`

## Slide 5: Reactive Navigation And Control

The controller maps the tracked gate estimate into body-rate/thrust:

```text
yaw_rate   <- horizontal bearing
roll_rate  <- horizontal bearing
thrust     <- vertical bearing + hover estimate
pitch_rate <- forward approach from range/confidence
```

When no gate is visible, the controller enters a slow yaw search instead of guessing from simulator truth.

## Slide 6: Backend Adapters

The algorithm boundary is:

```text
RacingCommand(roll_rate, pitch_rate, yaw_rate, thrust)
```

Adapters map that command to the runtime:

- Elodin/Betaflight: RC PWM fields
- VQ1/MAVSDK: attitude-rate/thrust fields

This keeps perception/control logic independent from simulator-specific packet formats.

## Slide 7: Local Validation Harness

Because official VQ1 access requires Windows and is not yet available locally, validation uses the Elodin AI Grand Prix practice harness.

What it gives us:

- local macOS execution
- real Betaflight SITL in the loop
- FPV editor view
- run database and trace export
- a better debugging environment than the old PyBullet track work

Important caveat:

Elodin exposes world pose, but the CS260C pilot does not pass world pose into `AutonomousRacingPilot`.

## Slide 8: Current Results

Current verified results:

- active algorithm package compiles
- synthetic gate overlay produces a valid detection and control command
- no-FPV Elodin Betaflight smoke test succeeds
- CS260C Elodin solver adapter succeeds in the same smoke test
- optional pilot CSV trace logs mode, confidence, bearings, command, and RC output

Commands:

```bash
scripts/inspect_gate_frames.py --demo --out-dir logs/gate_inspection_demo --save-mask
scripts/run_elodin_smoke.sh
RACE_SOLVER=solver.cs260c_pilot scripts/run_elodin_smoke.sh
scripts/run_elodin_editor.sh
```

## Slide 9: Limitations

Current limitations:

- HSV detector is provisional until real VQ1/Elodin FPV frames are tuned
- no finished VQ1 Windows runtime yet
- Elodin command path uses Betaflight RC, while official VQ1 likely needs MAVSDK/MAVLink
- current planner is completion-first, not a time-optimal racing-line planner

## Slide 10: Next Steps

Immediate next steps:

1. Collect real Elodin FPV frames.
2. Tune the detector and save overlays.
3. Add a small labeled frame set.
4. Implement VQ1 MAVSDK adapter when simulator credentials arrive.
5. Use logs from successful reactive runs as data for a compact learned policy.

## One-Sentence Close

The project now has a clean competition-facing algorithm: FPV gate recognition, short-horizon tracking, reactive navigation, and simulator-independent body-rate/thrust control.

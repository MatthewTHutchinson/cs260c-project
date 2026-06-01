# Final Presentation Deck Source

Date: 2026-06-01

This is the build-ready slide source. Keep slide text short; use the speaker notes for explanation.

## Slide 1: Autonomous Drone Gate Racing

Kicker: `PROBLEM`

Title: `A drone has to race gates from FPV and telemetry, without global position.`

On-slide:

- Inputs: FPV frames plus vehicle telemetry
- No GPS, no depth shortcut, no absolute world pose
- Output: pilot-style control commands
- VQ1 target: complete the course reliably

Visual:

- Simple pipeline diagram:

```text
FPV + telemetry -> gate estimate -> navigation mode -> body-rate/thrust command
```

Speaker note:

The first project decision is to optimize for successful autonomous completion, not fastest lap time. VQ1 is a completion-first round, so the system needs to be transparent and easy to debug.

## Slide 2: Algorithm Thesis

Kicker: `APPROACH`

Title: `The core algorithm is a visual-servoing autopilot with a clean command boundary.`

On-slide:

- Detect the next gate in image space
- Track short dropouts over time
- Convert bearing and range into body-rate/thrust
- Adapt the same command to Elodin now and MAVSDK later

Visual:

```text
GateDetector -> GateTracker -> ReactiveGateController -> CommandAdapter
```

Speaker note:

The important boundary is `RacingCommand(roll_rate, pitch_rate, yaw_rate, thrust)`. That keeps the competition-facing logic separate from simulator-specific details like Betaflight RC packets or MAVSDK message objects.

## Slide 3: Gate Recognition

Kicker: `PERCEPTION`

Title: `The detector turns a visible gate into bearing, range, and confidence.`

On-slide:

- HSV segmentation baseline
- Contour and rectangle extraction
- Bearing from camera intrinsics
- Range estimate from apparent gate width

Visual:

- `assets/presentation/gate_sequence_detected.jpg`

Speaker note:

This detector is intentionally simple. It is not the final answer for every environment, but it is transparent. If the gate is off-center, we can see the measured bearing and directly understand the yaw command.

## Slide 4: Temporal Tracking

Kicker: `TRACKING`

Title: `The tracker prevents one missed frame from resetting the controller.`

On-slide:

- `detected`: current frame has a gate
- `commit`: close enough to keep passing through
- `tracked`: recent estimate is still fresh
- `search`: detection has aged out

Visual:

- `assets/presentation/gate_sequence_trace.png`

Speaker note:

The blue mode line shows the state machine. The green confidence line decays after the gate disappears. This is the bridge between brittle frame-by-frame detection and stable control.

## Slide 5: Commit And Search Behavior

Kicker: `NAVIGATION`

Title: `The controller changes behavior as the gate becomes close or disappears.`

On-slide:

- Center the gate during approach
- Commit near the gate instead of dithering
- Use short-term memory after detection loss
- Fall back to deliberate yaw search

Visual:

- Left: `assets/presentation/gate_sequence_commit.jpg`
- Right: `assets/presentation/gate_sequence_search.jpg`

Speaker note:

In `commit`, the gate is close and centered, so commands become small and steady. After loss, the tracker briefly holds the last estimate; once confidence expires, the autopilot switches to search yaw.

## Slide 6: Control Law

Kicker: `CONTROL`

Title: `Image error maps directly into body-rate and thrust commands.`

On-slide:

```text
yaw_rate   <- horizontal bearing
roll_rate  <- horizontal bearing
thrust     <- vertical bearing + hover estimate
pitch_rate <- range and confidence
```

- Commands are clipped before leaving the algorithm
- Search mode commands a slow yaw scan

Visual:

- Use the command-rate and RC sections of `assets/presentation/gate_sequence_trace.png`

Speaker note:

This is not an optimal racing-line controller. It is a reliable completion baseline. Its value is that every command can be traced back to a perception measurement.

## Slide 7: Simulator Adapter Boundary

Kicker: `ADAPTERS`

Title: `Simulator-specific packets are isolated behind one command interface.`

On-slide:

- Internal: `RacingCommand`
- Elodin: Betaflight RC PWM fields
- VQ1 target: MAVSDK attitude-rate/thrust fields
- World pose is never passed into the algorithm

Visual:

```text
RacingCommand
  -> to_betaflight_rc_fields(...)
  -> to_mavsdk_attitude_rate_fields(...)
```

Speaker note:

This is the workaround for the Elodin caveats. Elodin uses Betaflight UDP and exposes world pose, while VQ1 uses MAVSDK-compatible MAVLink and no global position. The algorithm boundary keeps those concerns from contaminating perception and control.

## Slide 8: Local Validation Harness

Kicker: `VALIDATION`

Title: `Elodin gives us a better local harness while official VQ1 access is pending.`

On-slide:

- macOS-compatible practice harness
- Real Betaflight SITL in the loop
- FPV editor path for visual inspection
- No-FPV smoke path for reliable control checks
- CSV trace for mode, detection, command, and RC output

Visual:

```text
CS260C algorithm -> Elodin solver hook -> Betaflight SITL -> Elodin physics
```

Speaker note:

The old PyBullet work is now historical context. Elodin is not the official simulator, but it is higher quality for local debugging and matches the race-oriented workflow more closely.

## Slide 9: Current Evidence

Kicker: `RESULTS`

Title: `The current stack is runnable, inspectable, and ready for real FPV tuning.`

On-slide:

- Active algorithm package compiles
- Synthetic sequence exercises detected, commit, tracked, and search modes
- Elodin no-FPV Betaflight smoke test passes
- CS260C Elodin solver adapter passes the same smoke test
- Pilot trace logs mode, confidence, bearing, command, and RC fields

Visual:

- Small table with validation checks

Speaker note:

The result is not a claimed VQ1 completion yet. The result is a clean, inspectable control stack ready to tune on real simulator frames once VQ1 or the Elodin editor path is available.

## Slide 10: Next Steps

Kicker: `NEXT`

Title: `The next work is tuning on real frames and connecting MAVSDK.`

On-slide:

1. Capture Elodin and VQ1 FPV frames.
2. Tune detector thresholds and candidate scoring.
3. Add a small labeled frame set.
4. Implement the MAVSDK attitude-rate adapter.
5. Use reactive-run logs as data for a compact learned policy.

Visual:

```text
Reactive baseline -> logged attempts -> compact learned policy
```

Speaker note:

The learning path is still there, but it comes after the observable baseline works. The immediate project is a robust autonomous racing algorithm, not another opaque training run.

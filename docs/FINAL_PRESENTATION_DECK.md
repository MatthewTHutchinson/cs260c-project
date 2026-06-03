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
- Optional neural detector backend
- Contour and rectangle extraction
- Bearing from camera intrinsics
- Range estimate from apparent gate width

Visual:

- `assets/presentation/gate_sequence_detected.jpg`

Speaker note:

This detector is intentionally simple. It is not the final answer for every environment, but it is transparent. The detector boundary now also accepts neural corner, box, center/range, or segmentation outputs, which means the learned-CV upgrade does not require rewriting the tracker and controller.

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

Title: `The baseline completes simple courses and exposes the next hard problem.`

On-slide:

- Straight / lateral / height-varied local courses: complete
- 4-gate straight course: complete
- Circular arc: `2/4` gates before DNF
- S-curve: `1/5` gates before DNF
- Failure mode: reactive controller cuts corners without sequence-aware planning

Visual:

- Small table from `docs/FINAL_PRESENTATION_RESULTS.md`
- Optional trace plots:
  - `assets/presentation/circular_arc_trace.png`
  - `assets/presentation/s_curve_trace.png`

Speaker note:

The result is not "solved drone racing." It is a clean baseline with a useful
failure boundary. It can fly visible forward-progressing gates, but the curved
and S-shaped tracks expose why the next version needs gate sequencing,
lookahead, and either a learned policy or model-based trajectory controller.

## Slide 10: Next Steps

Kicker: `NEXT`

Title: `The next work is sequence-aware navigation and stronger perception.`

On-slide:

1. Add gate-sequence memory for multiple gates in view.
2. Add future-gate lookahead for curved tracks.
3. Test a pretrained or fine-tuned neural gate detector.
4. Implement the MAVSDK attitude-rate adapter.
5. Use reactive-run logs as data for a compact learned policy.

Visual:

```text
Reactive baseline -> logged attempts -> compact learned policy
```

Speaker note:

The learning path is now modular. First replace the detector if it improves gate candidates; then add sequence-aware candidate selection and lookahead; then consider replacing the reactive controller with MPC or a compact learned policy.

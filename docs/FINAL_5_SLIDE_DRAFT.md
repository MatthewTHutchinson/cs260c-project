# Final Presentation 5-Slide Draft

Date: 2026-06-03

Target: 5 slides, max 4.5 minutes. Keep the talk easy to speak: one claim per
slide, one visual proof object per slide, and no dense implementation dump.

## Talk Spine

```text
Problem
  -> My algorithm
  -> How perception/tracking becomes control
  -> What worked locally
  -> What failed and what that teaches us
```

The story is not "we restarted the project." The story is: this is an FPV-based
autonomous drone racing system with a clear perception, navigation, and control
pipeline. Keep the talk focused on the system, the evidence, and the next
algorithmic step.

## Slide 1: Problem And Constraint

Title: `Autonomous gate racing from FPV, without global position`

On-slide:

- Input: FPV camera plus telemetry
- No GPS, depth, simulator gate IDs, or world pose
- Output: body-rate and thrust commands
- Goal: complete VQ1-style courses reliably

Figure:

- Simple pipeline diagram:

```text
FPV + telemetry -> gate estimate -> tracker state -> body-rate/thrust command
```

Speaker note, about 45 seconds:

> This project is an FPV-based autonomous drone racing system. The input is
> camera frames plus telemetry, and the output is body-rate and thrust commands.
> The important constraint is that the system does not use global position, GPS,
> depth, simulator gate IDs, or pre-known gate coordinates. So the core problem
> is perception-driven navigation: detect the next gate, maintain a target over
> time, and control the drone through it.

## Slide 2: Algorithm Architecture

Title: `The system is a modular FPV-to-control autopilot`

On-slide:

- Perception estimates gate geometry from FPV frames
- Tracking preserves a target through dropped detections
- Navigation state chooses detected, tracked, commit, or search
- Control outputs roll-rate, pitch-rate, yaw-rate, and thrust

Figure:

```text
GateDetector -> GateTracker -> ReactiveGateController -> RacingCommand
                                                    -> runtime adapter
```

Speaker note, about 55 seconds:

> The system is split into perception, tracking, navigation state, and control.
> The detector turns each FPV frame into a gate estimate. The tracker adds short
> memory, so one missed frame does not immediately reset the system. The
> controller then converts that navigation state into roll rate, pitch rate, yaw
> rate, and thrust. That command boundary is important because the autonomy logic
> stays independent of whichever simulation or runtime adapter is used.

## Slide 3: Perception To Control

Title: `Tracking turns noisy detections into navigation modes`

On-slide:

- Detector: gate center, bearing, range, confidence
- Tracker: short memory over missed frames
- `commit`: continue through a near gate
- `search`: level, brake drift, then yaw scan

Figure:

- Use `assets/presentation/gate_sequence_trace.png`
- Optional inset: `assets/presentation/gate_sequence_detected.jpg`

Speaker note, about 60 seconds:

> The perception output is not used directly as a steering command. It first
> becomes a tracked navigation target. If the gate is visible, the mode is
> detected. If the detector drops out briefly, the tracker holds the target. When
> the gate is near, commit mode keeps the drone moving through instead of chasing
> clipped edges. If confidence expires, search mode levels the drone, brakes
> drift, and then yaw-scans for a new gate.

## Slide 4: Local Validation Results

Title: `Evidence: simple tracks work; curved tracks expose the gap`

On-slide:

| Course family | Result |
|---|---:|
| Straight, lateral, height-varied | Complete |
| Four-gate straight | Complete |
| Circular arc | 2/4 gates, DNF |
| S-curve | 1/5 gates, DNF |

Small evidence callout:

```text
Core suite: 4/4 courses complete
circular: 2/4 DNF
```

Figure:

- Results table as a compact native table.
- Optional small FPV image strip from:
  - `assets/presentation/gate_sequence_detected.jpg`
  - `assets/presentation/gate_sequence_commit.jpg`
  - `assets/presentation/gate_sequence_search.jpg`

Speaker note, about 50 seconds:

> The baseline completes the simple course suite: straight, mild lateral,
> height-varied, and four-gate straight tracks. That shows the perception,
> tracking, and control loop is functioning end to end. The failure cases are
> more informative: on the circular arc it reaches two of four gates, and on the
> S-curve it reaches one of five. So the current system is not just failing
> randomly; it fails when the next correct action depends on future course
> geometry.

## Slide 5: What Failed And Next Step

Title: `Next step: sequence-aware navigation plus learned control`

On-slide:

- Current controller reacts to the visible gate
- Curved tracks require sequence belief and lookahead
- Add a local corridor target or short-horizon planner
- Learn the policy with CNN visual features plus GRU/LSTM memory

Figure:

- Best figure: `assets/presentation/circular_arc_trace.png`
- Backup/appendix figure: `assets/presentation/s_curve_trace.png`

Speaker note, about 60 seconds:

> The main limitation is not just detection. The current controller can center a
> visible gate, but it does not reason about future course geometry. The next
> classical step is sequence-aware navigation: estimate which gate is next, infer
> a local corridor, and command a body-frame target with lookahead. The deep
> learning version I would test is a CNN plus GRU policy. The CNN extracts gate
> and visual-track features from recent FPV frames, the GRU keeps short-term
> memory through occlusions and fast turns, and a small MLP head outputs roll
> rate, pitch rate, yaw rate, and thrust.

## Recommended Figures

Use these:

1. `gate_sequence_trace.png`: best all-in-one proof for detector/tracker/mode
   behavior.
2. `gate_sequence_detected.jpg`: intuitive FPV visual for what the detector sees.
3. `circular_arc_trace.png`: best proof of the current failure boundary.
4. Small results table: easiest way to communicate evidence in 10 seconds.

Use video only if it does not steal time:

1. A short local-simulator preview GIF can work as silent background or quick
   visual context on Slide 1.
2. `legacy/pybullet/assets/presentation/state_champion_demo.mp4` is a
   9-second legacy PyBullet clip. Skip this in the main talk unless explicitly
  labeled as old work; it risks confusing the current simulation/VQ1 story.

Probably skip these in the main 5 slides:

- `s_curve_trace.png`: useful if asked, but circular arc is cleaner to explain.
- Legacy PyBullet video: visually nice, but it points at the old project rather
  than the current algorithm.
- Long code/pseudocode screenshots: too dense for 4.5 minutes.
- Full 10-slide architecture detail: accurate but too much for this timebox.

## Deep Learning Framing

Keep the deep learning next step concrete but not overclaimed:

```text
FPV frame
  -> compact CNN / neural gate detector
  -> gate features or image embedding
  -> GRU/LSTM temporal policy with telemetry
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

Recommended phrasing:

> The next learned controller I would test is a CNN plus GRU policy. The CNN
> extracts gate and visual-track features from recent FPV frames, the GRU keeps
> short-term memory through occlusions and fast turns, and a small MLP head
> outputs the same body-rate/thrust command used by the current controller.

Why this fits the class:

- It is a sequence model, not just frame-by-frame perception.
- It can learn recovery behavior from recent history.
- It still respects the current command boundary.
- It can be trained first from logged reactive runs, then improved with
  reinforcement learning in simulation.

## Timing Plan

```text
Slide 1: 0:45
Slide 2: 0:55
Slide 3: 1:00
Slide 4: 0:55
Slide 5: 1:05
Total:   4:40 if spoken slowly, target 4:20 with normal pacing
```

If the hard limit is strict, shorten Slide 5 by saying only: "The next deep
learning version is a CNN plus GRU policy that uses recent FPV frames and
telemetry to output the same body-rate and thrust command."

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

The story is not "we restarted the project." The story is: I built and tested a
competition-facing autonomous drone racing baseline that uses FPV perception,
temporal gate tracking, and body-rate/thrust commands without privileged
position.

## Slide 1: Problem And Constraint

Title: `Autonomous gate racing from FPV, without global position`

On-slide:

- Input: FPV camera plus telemetry
- No GPS, depth, simulator gate IDs, or world pose
- Output: pilot-style roll/pitch/yaw/thrust commands
- Goal: complete VQ1-style courses reliably

Figure:

- Simple pipeline diagram:

```text
FPV + telemetry -> gate estimate -> tracker state -> body-rate/thrust command
```

Speaker note, about 45 seconds:

> The task is autonomous drone racing, but the important constraint is that the
> algorithm cannot use global position. It sees the world through an FPV camera
> and telemetry, then has to output the same kind of pilot commands a controller
> would use. For this stage, I optimized for completion and debuggability rather
> than maximum speed.

## Slide 2: Algorithm Architecture

Title: `The system is a modular FPV-to-command autopilot`

On-slide:

- Classical gate detector now, neural detector backend later
- Temporal tracker handles dropped detections
- Reactive controller outputs body rates and thrust
- Same `RacingCommand` boundary maps to the local simulator now and MAVSDK later

Figure:

```text
GateDetector -> GateTracker -> ReactiveGateController -> RacingCommand
                                                    -> simulator / MAVSDK adapter
```

Speaker note, about 55 seconds:

> The key design choice is the command boundary. The algorithm produces
> `RacingCommand`: roll rate, pitch rate, yaw rate, and normalized thrust. That
> keeps the racing logic separate from the simulator adapter. Locally I map it
> to RC-style commands in the simulation environment, but the same boundary is
> intended to map to MAVSDK attitude-rate and thrust commands for VQ1.

## Slide 3: Perception To Control

Title: `A visible gate becomes bearing, range, confidence, and a mode`

On-slide:

- Detector estimates gate center, bearing, rough range, confidence
- Tracker modes: `detected`, `tracked`, `commit`, `search`
- `commit`: pass through instead of chasing clipped gate edges
- `search`: level, brake drift, then yaw scan

Figure:

- Use `assets/presentation/gate_sequence_trace.png`
- Optional inset: `assets/presentation/gate_sequence_detected.jpg`

Speaker note, about 60 seconds:

> The detector is intentionally inspectable. It segments gate-like colors,
> extracts contours, and estimates image bearing and rough range. The tracker is
> what makes this usable in flight: it smooths short missed detections, commits
> near the gate, and switches to search only after confidence expires. Search
> was important to fix because yawing while still pitched forward makes the
> drone scan off-axis, so the current version levels and brakes first.

## Slide 4: Local Validation Results

Title: `The baseline completes simple local courses`

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

Speaker note, about 55 seconds:

> In local simulation validation, the baseline completed the straight, mild lateral,
> height-varied, and four-gate straight courses. The latest core suite completed
> all four. But the same controller failed on harder curved courses: it got two
> of four gates on the circular arc and one of five on the S-curve. That is a
> useful result because it tells us exactly where the baseline stops working.

## Slide 5: What Failed And Next Step

Title: `The next missing layer is lookahead navigation`

On-slide:

- Current controller reacts to the visible gate
- Curved tracks require sequence belief and future-gate lookahead
- Next: local corridor target or short-horizon planner
- Next deep learning step: neural gate detector + recurrent learned controller

Figure:

- Best figure: `assets/presentation/circular_arc_trace.png`
- Backup/appendix figure: `assets/presentation/s_curve_trace.png`

Speaker note, about 65 seconds:

> The main limitation is not just detection. The controller can center a visible
> gate, but it does not understand the future course geometry. On a circular or
> S-shaped track, that means it can pass one gate and then fly wide or cut the
> wrong line. The next algorithmic layer should be sequence-aware navigation:
> track which gate is next, infer a local corridor, and command a body-frame
> target with lookahead. The deep learning extension is to replace the classical
> detector with a neural gate detector, then train a recurrent controller that
> maps recent FPV features and telemetry into body-rate and thrust commands.

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

# Final Presentation Results

Date: 2026-06-03

## Current Algorithm

```text
FPV frame
  -> classical HSV/contour gate detector
  -> distance-aware temporal gate tracker
  -> reactive visual-servo controller
  -> body-rate/thrust command
  -> local simulator RC adapter
```

This is a completion-first baseline. It does not use GPS, world pose, simulator
gate IDs, pre-known gate coordinates, depth, RL, MPC, or a learned CV model.
The codebase now has a GateNet/ONNX detector adapter, but the results below
were produced by the classical detector.

## Local Simulation Validation

| Course | Shape | Result | Pass Times | Interpretation |
|---|---:|---:|---|---|
| `easy` | 3 straight gates | `3/3 COMPLETE` | `4.92, 6.34, 7.57` | baseline completion works |
| `lateral_soft` | 3 slight lateral-offset gates | `3/3 COMPLETE` | `4.92, 6.33, 7.61` | mild lateral correction works |
| `low_high` | 3 height-varied gates | `3/3 COMPLETE` | `4.84, 6.20, 7.36` | camera pitch/sign handling works |
| `four_gate_straight` | 4 straight gates | `4/4 COMPLETE` | `4.87, 6.25, 7.45, 8.54` | repeated reacquisition works |
| `circular` / `circular_arc` | 4 yawed gates on a gentle arc | `2/4 DNF` | `5.49, 6.81, --, --` | flies wide after gate 1 without lookahead |
| `s_curve` | 5 yawed gates with linked lateral wiggles | `1/5 DNF` | `4.95, --, --, --, --` | misses first lateral wiggle |

Run artifacts are saved under `logs/`, with presentation-ready trace plots in
`assets/presentation/circular_arc_trace.png` and
`assets/presentation/s_curve_trace.png`.

## Camera Profile A/B

The current harness can compare camera assumptions without replacing the
autonomy stack:

```text
camera_profile=vq1_pinhole
camera_profile=gatenet_fisheye
```

The `gatenet_fisheye` profile uses a `120 deg` vertical FOV, `144 deg`
horizontal FOV, and a fisheye render effect. The detector/controller use an
effective focal length of `103.92 px` for bearing/range approximation.

| Camera Profile | Course | Result | Pass Times | Interpretation |
|---|---|---:|---|---|
| `vq1_pinhole` | `easy` | `3/3 COMPLETE` | `4.90, 6.27, 7.48` | default profile still works |
| `vq1_pinhole` | `lateral_soft` | `3/3 COMPLETE` | `4.89, 6.30, 7.57` | lateral correction still works |
| `vq1_pinhole` | `low_high` | `3/3 COMPLETE` | `4.86, 6.24, 7.42` | height handling still works |
| `vq1_pinhole` | `four_gate_straight` | `0/4 DNF` then `4/4 COMPLETE` on immediate rerun | rerun: `4.88, 6.27, 7.50, 8.62` | first row exposed run-to-run sensitivity near gate 0 |
| `gatenet_fisheye` | `easy` | `3/3 COMPLETE` | `4.94, 6.30, 7.50` | wide/fisheye render path works |
| `gatenet_fisheye` | `lateral_soft` | `2/3 DNF` | `4.92, 6.30, --` | widened view changes lateral behavior enough to miss gate 2 |
| `gatenet_fisheye` | `low_high` | `3/3 COMPLETE` | `4.92, 6.29, 7.48` | height-varied straight course still works |
| `gatenet_fisheye` | `four_gate_straight` | `4/4 COMPLETE` | `4.95, 6.36, 7.59, 8.72` | repeated straight reacquisition works |

Takeaway: the fisheye profile is viable as an experiment, but it does not
magically fix navigation. With the current classical detector and reactive
controller, wider FOV helps some reacquisition cases and hurts at least one
lateral-offset case. The next step is still sequence-aware candidate selection
and then an actual GateNet/ONNX detector.

## Takeaway

The current stack is a strong transparent baseline for visible, mostly
forward-progressing courses. It fails on curved and S-shaped courses because
its sequence belief is still only a conservative visual proxy, with no
future-gate lookahead and no trajectory planner. It sees the current
largest/nearest gate-like contour, adds short post-pass stale-candidate
rejection, and reacts locally.

This is the right motivation for the next algorithm layer:

- explicit navigation state and sequence belief
- future-gate lookahead
- stronger perception for partial/multiple gates
- learned policy or MPC for dynamics-aware turning

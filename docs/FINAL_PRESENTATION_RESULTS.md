# Final Presentation Results

Date: 2026-06-03

## Current Algorithm

```text
FPV frame
  -> classical HSV/contour gate detector
  -> temporal gate tracker
  -> reactive visual-servo controller
  -> body-rate/thrust command
  -> Elodin Betaflight RC adapter
```

This is a completion-first baseline. It does not use GPS, world pose, simulator
gate IDs, pre-known gate coordinates, depth, RL, MPC, or a learned CV model.
The codebase now has an optional neural detector adapter, but the results below
were produced by the classical detector.

## Local Elodin Validation

| Course | Shape | Result | Pass Times | Interpretation |
|---|---:|---:|---|---|
| `easy` | 3 straight gates | `3/3 COMPLETE` | `4.90, 6.26, 7.47` | baseline completion works |
| `lateral_soft` | 3 slight lateral-offset gates | `3/3 COMPLETE` | `4.88, 6.26, 7.48` | mild lateral correction works |
| `low_high` | 3 height-varied gates | `3/3 COMPLETE` | `4.90, 6.25, 7.42` | camera pitch/sign handling works |
| `four_gate_straight` | 4 straight gates | `4/4 COMPLETE` | `4.90, 6.31, 7.54, 8.69` | repeated reacquisition works |
| `circular_arc` | 4 yawed gates on a gentle arc | `2/4 DNF` | `5.54, 6.86, --, --` | cuts inside the curve after gate 1 |
| `s_curve` | 5 yawed gates with linked lateral wiggles | `1/5 DNF` | `4.95, --, --, --, --` | misses first lateral wiggle |

Logs:

```text
logs/elodin_course_suite/summary.csv
logs/elodin_hard_track_results/summary.csv
assets/presentation/circular_arc_trace.png
assets/presentation/s_curve_trace.png
```

## Takeaway

The current stack is a strong transparent baseline for visible, mostly
forward-progressing courses. It fails on curved and S-shaped courses because it
has no explicit gate-sequence belief, no future-gate lookahead, and no
trajectory planner. It sees the current largest/nearest gate-like contour and
reacts locally.

This is the right motivation for the next algorithm layer:

- sequence-aware gate tracking
- future-gate lookahead
- stronger perception for partial/multiple gates
- learned policy or MPC for dynamics-aware turning

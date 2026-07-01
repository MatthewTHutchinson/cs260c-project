# Final Archived Artifacts

These files are the compact, non-regenerable subset of the ignored `logs/`
directory preserved when the project was archived on 2026-07-01.

## Learning Artifacts

- `teacher_curve_stress_rejoin_corridor.csv`: final 34-course privileged-teacher
  dataset with launch, nominal, off-nominal, and corridor phases.
- `feature_policy_curve_stress_corridor.pt`: final PyTorch checkpoint.
- `feature_policy_curve_stress_corridor.npz`: NumPy runtime export of the same
  policy for the Elodin solver.

## Closed-Loop Evidence

- `reactive_hard_tracks_summary.csv`: reactive baseline results for circular arc
  and S-curve.
- `reactive_circular_arc_trace.csv`: circular-arc reactive-controller trace.
- `reactive_s_curve_trace.csv`: S-curve reactive-controller trace.
- `corridor_student_s_curve_summary.csv`: short corridor-checkpoint rollout
  summary.
- `corridor_student_s_curve_trace.csv`: corresponding rollout trace. The trace
  records `command_source` and shows that the supervisor frequently selected
  `reactive_fallback`; it must not be presented as a pure student-policy run.

Large FPV frame dumps and repeated intermediate experiments were intentionally
excluded. See `docs/ARCHIVE_AND_RESTART.md` for restore commands and caveats.

# Due-Tonight Roadmap Status

Date: 2026-06-10

## Short Answer

The project is currently at **Step 4: DAgger-style closed-loop relabeling**, but
Step 4 is only partially complete.

Steps 1-3 are complete enough to report as project contributions. Step 4 is
implemented and has evidence, but it has not yet produced reliable closed-loop
course completion. Step 5, PPO/RL fine-tuning, should be framed as future work
because the simulator loop is not trustworthy enough to spend the final hours on
RL.

## Status By Roadmap Step

| Step | Status | Evidence | Report Claim |
| --- | --- | --- | --- |
| 1. Stop trusting weak teacher data | Done | Old reactive traces are treated as smoke tests only; the training path now uses generated privileged-teacher traces and a fail-fast teacher quality gate. | The project identified weak demonstrations as a failure mode and stopped training from them as the main source of truth. |
| 2. Build a privileged teacher, keep privilege out of policy | Done for offline BC | `scripts/generate_privileged_teacher_dataset.py` generates sequence-aware `rejoin` teacher labels from debug gate geometry; `scripts/audit_teacher_quality_gate.py` verifies quality; selected student features exclude privileged, sequence, and previous-command inputs. | Privileged simulator geometry is used only for labels/debug fields, while the deployed GRU consumes legal FPV-derived features and telemetry-like state. |
| 3. Use classical CV as feature extractor | Done | The current feature policy consumes 22 legal detector/tracker/telemetry features: confidence, bearing, range, pixel center, apparent size, gate age, mode one-hot, deltas, and body velocity/elevation. | Classical CV is the inspectable perception frontend; the learned component is the temporal control policy. |
| 4. DAgger | Partially done | Closed-loop failure traces can be relabeled with `scripts/relabel_closed_loop_trace.py`; 3-relabel training reached the best closed-loop result so far, `DNF gates=1/3` on easy. Fresh offline rejoin BC remains strong, but closed-loop racing is not solved. | DAgger-style data aggregation is implemented and improved the evidence trail, but closed-loop recovery remains the bottleneck. |
| 5. PPO/RL fine-tuning | Not done | PPO needs a stable simulator loop and a BC/DAgger policy that can complete simple courses. Current closed-loop behavior is still too brittle. | PPO is the correct next research step after BC/DAgger stabilizes; it is intentionally not presented as a completed experiment. |

## Fresh Due-Tonight Experiment

Command:

```bash
scripts/run_feature_policy_ablation.sh
```

This regenerated the current rejoin-teacher dataset, passed the legal-feature
and teacher-quality gates, trained the no-prev/no-sequence GRU policy for 20
epochs, exported `.npz`, smoke-loaded the controller, and compared learned
commands against the reactive baseline on the held-out S-curve.

Teacher dataset:

```text
rows=14280
courses=26
randomized_courses=s_curve=12 arc=8
phases=launch:624,nominal:11160,off_nominal:2496
command_sources=teacher:14280
lookahead_forward_p1_m=0.465
backward_pct=0.00
max_gate_center_error_m=0.111
off_nominal_yaw_target_alignment_pct=100.0
selected_student_features=no_prev_command,no_sequence,no_privileged
feature_count=22
```

GRU BC result:

```text
best_val_mse=0.00244467
heldout_s_curve_mse=0.00185871
mae_roll_rate=0.030135
mae_pitch_rate=0.012719
mae_yaw_rate=0.053351
mae_thrust=0.007200
```

Learned vs reactive on held-out S-curve:

```text
learned_vs_teacher mse=0.00173593
reactive_vs_teacher mse=0.14797705

phase=nominal:
  learned_vs_teacher mse=0.00114335
  reactive_vs_teacher mse=0.17836463

phase=off_nominal:
  learned_vs_teacher mse=0.00490683
  reactive_vs_teacher mse=0.04058133
```

Interpretation:

- The learned GRU policy imitates the privileged rejoin teacher far better than
  the reactive visual-servo baseline on the held-out S-curve.
- The result is honest because the policy uses no previous-command features, no
  perfect sequence labels, and no privileged world/gate fields.
- The remaining gap is closed-loop robustness, not offline BC accuracy.

Useful figures:

```text
logs/learning_smoke/feature_bc_augmented_rejoin_no_prev_no_seq_leave_s_curve_out_20e_s_curve_audit.png
logs/controller_comparison/s_curve_rejoin_no_prev_no_seq_npz_comparison.png
assets/presentation/teacher_racing_lines_base.png
assets/presentation/teacher_racing_lines_full.png
```

The teacher racing-line plots show the nominal privileged teacher path, gates,
lookahead targets, launch samples, and off-nominal recovery samples. The base
plot marks `s_curve` as the held-out test course; the full plot shows the
randomized S-curve and arc augmentation courses used for training diversity.

Current gate-normal status: gate yaw/normal information is logged and visualized
with arrows, but the current `rejoin` teacher is primarily a gate-center and
path-tangent reference. It does not yet impose a hard "cross the gate plane from
the normal-facing side" constraint. That is acceptable for the current monotonic
debug tracks, where the centerline already moves forward through the gates, but
future minimum-snap/MPPI teacher upgrades should explicitly constrain gate plane
crossing direction.

## Search-Mode Fix

Gemini's search-mode braking change is kept. The controller now prioritizes
velocity braking before pitch leveling during search. This matters because
yaw-search while drifting forward can turn a lost-gate state into a large miss.

The repo audit now checks the combined case:

```text
braking_priority_search_command=expected_pitch=0.175 pitch=0.175 yaw=0.000
verdict=PASS
```

## What To Claim In The Report

Claim:

- This project built an autonomous drone racing pipeline with inspectable CV
  features and a deep temporal control policy.
- The key deep-learning result is a GRU behavior-cloning policy trained from a
  privileged sequence-aware teacher while preserving a legal deployed-input
  boundary.
- The learned policy strongly outperforms the hand-tuned reactive controller in
  offline command imitation on held-out curved/S-shaped tracks.
- DAgger-style relabeling is implemented and revealed the main bottleneck:
  closed-loop recovery after drift, especially post-gate reacquisition.

Do not claim:

- PPO was completed.
- The learned policy reliably completes all closed-loop courses.
- The local practice tracks are official VQ1 tracks.
- The policy consumes true gate IDs, GPS, world pose, or simulator gate centers.

## Next Best Work After Submission

1. Add a supervisor/state split for first-gate approach versus post-gate
   reacquisition.
2. Collect more closed-loop failures under the learned policy.
3. Relabel those states with the privileged rejoin/minimum-snap teacher.
4. Retrain and evaluate with closed-loop completion as the primary metric.
5. Start PPO only after BC/DAgger completes simple courses reliably.

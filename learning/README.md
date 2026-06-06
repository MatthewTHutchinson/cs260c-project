# Feature Policy Learning

This package is the first learning scaffold for the active FPV racing stack.
It is intentionally feature-based:

```text
classical CV gate features + telemetry + tracker history
  -> GRU/MLP policy
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

It does not train a raw-image CNN yet, and it does not use global pose,
simulator gate IDs, or known gate coordinates as policy inputs.

## Important Caveat

The current local trace CSVs prove the training plumbing works. They should not
be treated as high-quality racing demonstrations.

Known teacher problems:

- the visual-servo controller can bias toward gate edges/corners when detections
  are clipped or off-center
- lateral displacement and curved-track turns are weak
- circular and S-shaped courses expose the missing lookahead/navigation layer

Behavioral cloning will copy those mistakes. Use existing traces for smoke
tests and debugging only until better teacher labels are generated.

## T4 Setup

From the repo root on the GCP VM:

```bash
conda activate cs260c-t4
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Smoke Test

Overfit a synthetic trace:

```bash
python -m learning.train_bc \
  --demo-synthetic \
  --epochs 20 \
  --batch-size 64 \
  --out checkpoints/feature_bc_synthetic.pt
```

Evaluate the checkpoint:

```bash
python -m learning.eval_policy \
  --checkpoint checkpoints/feature_bc_synthetic.pt \
  --traces logs/learning_synthetic/trace.csv
```

## Local Trace Smoke Training

Point the same trainer at trace CSVs only to verify the loader/trainer on real
columns:

```bash
python -m learning.train_bc \
  --traces logs/elodin_course_suite \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_local_traces.pt
```

The loader recursively finds `trace.csv` files under directories.

Do not present this checkpoint as a capable autonomy policy. It is a plumbing
checkpoint.

## Data We Still Want

For stronger BC/DAgger training, future trace rows should include:

- detector/tracker features: bearing, range, confidence, mode, age
- telemetry: attitude, angular rates, linear velocity, acceleration when available
- previous command
- teacher command or local target command
- run metadata: course, camera profile, pass/fail status

The useful next labels should come from a better teacher:

- sequence-aware local corridor target
- minimum-snap or smooth lookahead reference for known debug courses
- improved hand-controller that targets the gate centerline and handles lateral
  displacement before we clone it

## Privileged Teacher Dataset

Generate a first privileged-teacher dataset from known debug course geometry:

```bash
python scripts/generate_privileged_teacher_dataset.py \
  --out logs/privileged_teacher/trace.csv
```

Audit the teacher before training on it:

```bash
python scripts/audit_privileged_teacher_dataset.py \
  --trace logs/privileged_teacher/trace.csv \
  --plot \
  --out-dir logs/privileged_teacher/audit
```

Train on it:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace.csv \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_privileged_teacher.pt
```

Evaluate it with course and mode breakdowns:

```bash
python -m learning.eval_policy \
  --checkpoint checkpoints/feature_bc_privileged_teacher.pt \
  --traces logs/privileged_teacher/trace.csv \
  --predictions-out logs/learning_smoke/feature_bc_privileged_teacher_predictions.csv
```

Train/evaluate a held-out course split:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace.csv \
  --exclude-courses s_curve \
  --epochs 20 \
  --batch-size 128 \
  --out logs/learning_smoke/feature_bc_leave_s_curve_out_20e.pt

python -m learning.eval_policy \
  --checkpoint logs/learning_smoke/feature_bc_leave_s_curve_out_20e.pt \
  --traces logs/privileged_teacher/trace.csv \
  --include-courses s_curve \
  --predictions-out logs/learning_smoke/feature_bc_leave_s_curve_out_20e_s_curve_predictions.csv
```

Generate randomized curved/S-shaped teacher variants:

```bash
python scripts/generate_privileged_teacher_dataset.py \
  --out logs/privileged_teacher/trace_with_variants.csv \
  --random-s-curve-variants 12 \
  --random-arc-variants 8 \
  --random-seed 11
```

Generate the current augmented teacher dataset with launch and off-nominal
recovery episodes:

```bash
python scripts/generate_privileged_teacher_dataset.py \
  --out logs/privileged_teacher/trace_augmented.csv \
  --random-s-curve-variants 12 \
  --random-arc-variants 8 \
  --launch-samples 24 \
  --off-nominal-episodes-per-course 4 \
  --off-nominal-length 24 \
  --random-seed 17
```

Rows are split by `episode_id`, so GRU windows do not cross between nominal,
launch, and recovery sequences. Startup windows are padded the same way the
runtime controller pads its first few frames.

Train on those variants while still holding out the canonical `s_curve`:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace_with_variants.csv \
  --exclude-courses s_curve \
  --epochs 20 \
  --batch-size 256 \
  --no-prev-command-features \
  --out logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.pt

python -m learning.eval_policy \
  --checkpoint logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.pt \
  --traces logs/privileged_teacher/trace_with_variants.csv \
  --include-courses s_curve \
  --predictions-out logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev_s_curve_predictions.csv
```

Run the stricter no-sequence ablation before spending real T4 time. This drops
`last_gate_passed` and `next_gate_index`, which are tracker beliefs at runtime
but can be unrealistically perfect in privileged teacher data:

Regenerate `trace_augmented.csv` first if it predates the `frame_fresh` column;
the audit below intentionally fails when selected trace features are missing.

```bash
python scripts/audit_learning_feature_spec.py \
  --no-prev-command-features \
  --no-sequence-features \
  --expect-no-prev-command-features \
  --expect-no-sequence-features \
  --trace logs/privileged_teacher/trace_augmented.csv

python -m learning.train_bc \
  --traces logs/privileged_teacher/trace_augmented.csv \
  --exclude-courses s_curve \
  --epochs 20 \
  --batch-size 256 \
  --no-prev-command-features \
  --no-sequence-features \
  --out logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.pt

python -m learning.eval_policy \
  --checkpoint logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.pt \
  --traces logs/privileged_teacher/trace_augmented.csv \
  --include-courses s_curve \
  --predictions-out logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e_s_curve_predictions.csv

python -m learning.export_policy_npz \
  --checkpoint logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.pt \
  --out logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.npz
```

Export that checkpoint to a pure NumPy runtime artifact before deploying it in
the simulator harness:

```bash
python -m learning.export_policy_npz \
  --checkpoint logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.pt \
  --out logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.npz
```

The older teacher-forced version is useful for comparison, but should not be
the first runtime checkpoint:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace_with_variants.csv \
  --exclude-courses s_curve \
  --epochs 20 \
  --batch-size 256 \
  --out logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e.pt

python -m learning.eval_policy \
  --checkpoint logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e.pt \
  --traces logs/privileged_teacher/trace_with_variants.csv \
  --include-courses s_curve \
  --predictions-out logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_s_curve_predictions.csv
```

The loader prefers `teacher_*` target columns when they exist. It does not use
`world_*` or `teacher_next_gate_*` columns as student inputs.

The current privileged teacher uses a smooth cubic Hermite reference through the
gate centers and selects a future point on that reference as the lookahead
target. It is intentionally a stepping stone toward minimum-snap/reference
trajectory labels, not the deployed competition policy.

Current local 20-epoch smoke result, 2026-06-05:

```text
overall mse=0.00044975
easy mse=0.00004778
circular_arc mse=0.00039825
s_curve mse=0.00130725
```

Current leave-`s_curve`-out result:

```text
s_curve held-out mse=0.16744989
mae_yaw_rate=0.52123985
```

Current leave-`s_curve`-out result after adding 12 randomized S-curve variants
and 8 randomized arc variants, with previous-command features:

```text
s_curve held-out mse=0.00024772
mae_yaw_rate=0.01774498
```

Current runtime-friendlier no-previous-command checkpoint:

```text
s_curve held-out mse=0.00101455
mae_yaw_rate=0.03836820
runtime smoke thrust_norm=0.557330
```

The model can fit the upgraded teacher when S-curve examples are included, but
it does not generalize to S-curves from straight/soft-curve examples alone.
Randomized curved teacher data fixes the first generalization failure much more
effectively than simply increasing model complexity. For runtime rollout, prefer
the no-previous-command checkpoint until we have DAgger or closed-loop training
data; the teacher-forced previous-command model has lower offline MSE but fails
the replay smoke due to previous-command feedback mismatch.

## Runtime Smoke

After training a checkpoint, verify that it can be loaded by the competition
package and converted into a clipped `RacingCommand`:

```bash
python scripts/smoke_learned_controller.py \
  --checkpoint logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.npz \
  --trace logs/privileged_teacher/trace_with_variants.csv \
  --course s_curve \
  --rows 24
```

Audit prediction calibration:

```bash
python scripts/audit_policy_predictions.py \
  --predictions logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev_s_curve_predictions.csv \
  --plot logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev_s_curve_audit.png
```

Compare learned, reactive, and teacher commands before simulator rollout:

```bash
python scripts/compare_controllers_on_trace.py \
  --trace logs/privileged_teacher/trace_with_variants.csv \
  --checkpoint logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.npz \
  --course s_curve \
  --out logs/controller_comparison/s_curve_no_prev_npz_comparison.csv \
  --plot logs/controller_comparison/s_curve_no_prev_npz_comparison.png
```

Current S-curve comparison:

```text
learned_vs_teacher mse=0.00116282
reactive_vs_teacher mse=0.17836463
```

Current augmented/padded smoke result:

```text
dataset=logs/privileged_teacher/trace_augmented.csv
rows=14280
courses=26
checkpoint=logs/learning_smoke/feature_bc_augmented_padded_leave_s_curve_out_20e_no_prev.npz
heldout_s_curve_mse=0.02419935
comparison_s_curve learned_vs_teacher mse=0.02409630
comparison_s_curve reactive_vs_teacher mse=0.17126263
phase=launch learned_vs_teacher mse=0.00065725
phase=nominal learned_vs_teacher mse=0.00175707
phase=off_nominal learned_vs_teacher mse=0.13467118
```

The augmented labels fixed the first closed-loop failure mode: launch/thrust.
In a 10 s `easy` Elodin smoke, first learned thrust increased from `0.300` to
`0.722`, and the drone climbed to racing height. It still missed gate 0
laterally, so the next labels should come from closed-loop failure traces with
privileged debug world position, then relabeled back to the same legal feature
inputs.

## Closed-Loop Relabeling

Relabel a failed Elodin rollout after applying the debug-world trace patch:

```bash
python scripts/relabel_closed_loop_trace.py \
  --trace logs/elodin_learned_relabel_seed/vq1_pinhole/easy/trace.csv \
  --course easy \
  --episode-id easy:closed_loop:relabel_seed_001 \
  --out logs/privileged_teacher/closed_loop_relabels/easy_relabel_seed_001.csv

python scripts/audit_closed_loop_signs.py \
  --trace logs/elodin_learned_relabel_seed/vq1_pinhole/easy/trace.csv \
  --course easy
```

Train with the nominal/augmented teacher plus the relabeled failure episode:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace_augmented.csv \
           logs/privileged_teacher/closed_loop_relabels/easy_relabel_seed_001.csv \
  --exclude-courses s_curve \
  --epochs 20 \
  --batch-size 256 \
  --no-prev-command-features \
  --out logs/learning_smoke/feature_bc_augmented_plus_relabel_20e_no_prev.pt

python -m learning.export_policy_npz \
  --checkpoint logs/learning_smoke/feature_bc_augmented_plus_relabel_20e_no_prev.pt \
  --out logs/learning_smoke/feature_bc_augmented_plus_relabel_20e_no_prev.npz
```

First relabel iteration, 2026-06-04:

```text
closed_loop_relabel_rows=383
offline_relabel_fit learned_vs_teacher mse=0.00120523
8s_easy_rollout status=DNF gates=0/3
end_lateral_error_m=13.818540
mean_thrust=0.547822
corr_roll_to_abs_error_rate=-0.437697
corr_yaw_to_abs_error_rate=0.217393
```

Interpretation: the DAgger loop is wired, and roll is now directionally helpful
on the audited rollout, but yaw/altitude recovery are still wrong enough that
the learned controller misses right and flies low. The next relabeling teacher
should reduce yaw authority during large lateral errors and raise recovery
thrust/altitude hold before another T4-scale training run.

The runtime wrapper lives in `algorithm/learned_controller.py`. It is optional:
`AutonomousRacingPilot` uses it only when one is supplied and the current gate
estimate is usable, otherwise the reactive controller handles search/fallback.

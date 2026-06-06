#!/usr/bin/env bash
# Rebuild the current feature-policy baseline without previous-command or
# perfect-sequence inputs, then run the offline audits used for project results.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PROJECT_PYTHON:-/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python}"

# macOS conda can load duplicate OpenMP runtimes when torch/matplotlib/numpy are
# imported together. Linux/T4 runs normally with this value unused.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

TRACE="${TRACE:-logs/privileged_teacher/trace_augmented.csv}"
COURSE="${COURSE:-s_curve}"
CHECKPOINT="${CHECKPOINT:-logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.pt}"
NPZ="${NPZ:-logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.npz}"
PREDICTIONS="${PREDICTIONS:-logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e_s_curve_predictions.csv}"
PREDICTION_PLOT="${PREDICTION_PLOT:-logs/learning_smoke/feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e_s_curve_audit.png}"
COMPARISON="${COMPARISON:-logs/controller_comparison/s_curve_no_prev_no_seq_npz_comparison.csv}"
COMPARISON_PLOT="${COMPARISON_PLOT:-logs/controller_comparison/s_curve_no_prev_no_seq_npz_comparison.png}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo "[CS260C] python=$PYTHON_BIN"
echo "[CS260C] trace=$TRACE"
echo "[CS260C] checkpoint=$CHECKPOINT"

"$PYTHON_BIN" scripts/generate_privileged_teacher_dataset.py \
  --out "$TRACE" \
  --random-s-curve-variants 12 \
  --random-arc-variants 8 \
  --launch-samples 24 \
  --off-nominal-episodes-per-course 4 \
  --off-nominal-length 24 \
  --random-seed 17

"$PYTHON_BIN" scripts/audit_learning_feature_spec.py \
  --no-prev-command-features \
  --no-sequence-features \
  --expect-no-prev-command-features \
  --expect-no-sequence-features \
  --trace "$TRACE"

"$PYTHON_BIN" -m learning.train_bc \
  --traces "$TRACE" \
  --exclude-courses "$COURSE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --no-prev-command-features \
  --no-sequence-features \
  --out "$CHECKPOINT"

"$PYTHON_BIN" -m learning.eval_policy \
  --checkpoint "$CHECKPOINT" \
  --traces "$TRACE" \
  --include-courses "$COURSE" \
  --predictions-out "$PREDICTIONS"

"$PYTHON_BIN" -m learning.export_policy_npz \
  --checkpoint "$CHECKPOINT" \
  --out "$NPZ"

"$PYTHON_BIN" scripts/audit_learning_feature_spec.py \
  --checkpoint "$NPZ" \
  --expect-no-prev-command-features \
  --expect-no-sequence-features

"$PYTHON_BIN" scripts/smoke_learned_controller.py \
  --checkpoint "$NPZ" \
  --trace "$TRACE" \
  --course "$COURSE" \
  --rows 24

"$PYTHON_BIN" scripts/audit_policy_predictions.py \
  --predictions "$PREDICTIONS" \
  --plot "$PREDICTION_PLOT"

"$PYTHON_BIN" scripts/compare_controllers_on_trace.py \
  --trace "$TRACE" \
  --checkpoint "$NPZ" \
  --course "$COURSE" \
  --out "$COMPARISON" \
  --plot "$COMPARISON_PLOT"

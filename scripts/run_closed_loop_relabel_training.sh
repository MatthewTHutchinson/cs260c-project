#!/usr/bin/env bash
# Train the deployable feature policy on the rejoin teacher plus one or more
# closed-loop relabel CSVs. This is the lightweight DAgger-style iteration loop.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PROJECT_PYTHON:-/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python}"

# macOS conda can load duplicate OpenMP runtimes when torch/numpy/matplotlib mix.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASE_TRACE="${BASE_TRACE:-logs/privileged_teacher/trace_augmented_rejoin.csv}"
RELABELS="${RELABELS:-}"
COURSE="${COURSE:-s_curve}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_NAME="${RUN_NAME:-feature_bc_augmented_rejoin_plus_relabels_no_prev_no_seq_20e}"
CHECKPOINT="${CHECKPOINT:-logs/learning_smoke/${RUN_NAME}.pt}"
NPZ="${NPZ:-logs/learning_smoke/${RUN_NAME}.npz}"
PREDICTIONS="${PREDICTIONS:-logs/learning_smoke/${RUN_NAME}_${COURSE}_predictions.csv}"
RELABEL_PREDICTIONS="${RELABEL_PREDICTIONS:-logs/learning_smoke/${RUN_NAME}_relabel_predictions.csv}"

if [[ ! -f "$BASE_TRACE" ]]; then
  echo "[CS260C] missing BASE_TRACE=$BASE_TRACE" >&2
  exit 1
fi

TRACE_ARGS=(--traces "$BASE_TRACE")
RELABEL_ARGS=()
if [[ -n "$RELABELS" ]]; then
  # shellcheck disable=SC2206
  RELABEL_ARRAY=($RELABELS)
  for relabel in "${RELABEL_ARRAY[@]}"; do
    if [[ ! -f "$relabel" ]]; then
      echo "[CS260C] missing relabel=$relabel" >&2
      exit 1
    fi
    TRACE_ARGS+=("$relabel")
    RELABEL_ARGS+=("$relabel")
  done
fi

echo "[CS260C] python=$PYTHON_BIN"
echo "[CS260C] base_trace=$BASE_TRACE"
echo "[CS260C] relabel_count=${#RELABEL_ARGS[@]}"
echo "[CS260C] checkpoint=$CHECKPOINT"

"$PYTHON_BIN" scripts/audit_learning_feature_spec.py \
  --no-prev-command-features \
  --no-sequence-features \
  --expect-no-prev-command-features \
  --expect-no-sequence-features \
  --trace "$BASE_TRACE"

"$PYTHON_BIN" -m learning.train_bc \
  "${TRACE_ARGS[@]}" \
  --exclude-courses "$COURSE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --no-prev-command-features \
  --no-sequence-features \
  --out "$CHECKPOINT"

"$PYTHON_BIN" -m learning.eval_policy \
  --checkpoint "$CHECKPOINT" \
  --traces "$BASE_TRACE" \
  --include-courses "$COURSE" \
  --predictions-out "$PREDICTIONS"

if [[ ${#RELABEL_ARGS[@]} -gt 0 ]]; then
  "$PYTHON_BIN" -m learning.eval_policy \
    --checkpoint "$CHECKPOINT" \
    --traces "${RELABEL_ARGS[@]}" \
    --predictions-out "$RELABEL_PREDICTIONS"
fi

"$PYTHON_BIN" -m learning.export_policy_npz \
  --checkpoint "$CHECKPOINT" \
  --out "$NPZ"

"$PYTHON_BIN" scripts/audit_learning_feature_spec.py \
  --checkpoint "$NPZ" \
  --expect-no-prev-command-features \
  --expect-no-sequence-features

echo "[CS260C] npz=$NPZ"

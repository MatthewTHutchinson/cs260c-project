#!/usr/bin/env bash
# Train/evaluate the feature policy on a harder curved-track curriculum.
#
# This keeps the same legal-input GRU setup as run_feature_policy_ablation.sh,
# but adds generated hard S-curve courses and leaves one hard course out as the
# test split. Use this after the base rejoin experiment passes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TRACE="${TRACE:-logs/privileged_teacher/trace_curve_stress_rejoin.csv}"
export COURSE="${COURSE:-hard_s_curve_rand_000}"
export CHECKPOINT="${CHECKPOINT:-logs/learning_smoke/feature_bc_curve_stress_rejoin_no_prev_no_seq_leave_hard_s_curve_out_20e.pt}"
export NPZ="${NPZ:-logs/learning_smoke/feature_bc_curve_stress_rejoin_no_prev_no_seq_leave_hard_s_curve_out_20e.npz}"
export PREDICTIONS="${PREDICTIONS:-logs/learning_smoke/feature_bc_curve_stress_rejoin_no_prev_no_seq_leave_hard_s_curve_out_20e_predictions.csv}"
export PREDICTION_PLOT="${PREDICTION_PLOT:-logs/learning_smoke/feature_bc_curve_stress_rejoin_no_prev_no_seq_leave_hard_s_curve_out_20e_audit.png}"
export COMPARISON="${COMPARISON:-logs/controller_comparison/hard_s_curve_rejoin_no_prev_no_seq_npz_comparison.csv}"
export COMPARISON_PLOT="${COMPARISON_PLOT:-logs/controller_comparison/hard_s_curve_rejoin_no_prev_no_seq_npz_comparison.png}"
export RANDOM_S_CURVE_VARIANTS="${RANDOM_S_CURVE_VARIANTS:-12}"
export RANDOM_HARD_S_CURVE_VARIANTS="${RANDOM_HARD_S_CURVE_VARIANTS:-8}"
export RANDOM_ARC_VARIANTS="${RANDOM_ARC_VARIANTS:-8}"

# Hard curves deliberately push roll/yaw closer to saturation. This cap still
# catches broken labels while allowing the stress curriculum to exist.
export MAX_COMMAND_SATURATION_PCT="${MAX_COMMAND_SATURATION_PCT:-70}"

scripts/run_feature_policy_ablation.sh

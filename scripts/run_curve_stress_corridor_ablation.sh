#!/usr/bin/env bash
# Train/evaluate the hard-curve policy with explicit first-gate corridor
# recovery examples. This is the next BC/DAgger stabilization step after the
# first relabel checkpoint improved offline loss but missed gate 0 in closed
# loop.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TRACE="${TRACE:-logs/privileged_teacher/trace_curve_stress_rejoin_corridor.csv}"
export COURSE="${COURSE:-hard_s_curve_rand_000}"
export CHECKPOINT="${CHECKPOINT:-logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e.pt}"
export NPZ="${NPZ:-logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e.npz}"
export PREDICTIONS="${PREDICTIONS:-logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e_predictions.csv}"
export PREDICTION_PLOT="${PREDICTION_PLOT:-logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e_audit.png}"
export COMPARISON="${COMPARISON:-logs/controller_comparison/hard_s_curve_rejoin_corridor_no_prev_no_seq_npz_comparison.csv}"
export COMPARISON_PLOT="${COMPARISON_PLOT:-logs/controller_comparison/hard_s_curve_rejoin_corridor_no_prev_no_seq_npz_comparison.png}"

export RANDOM_S_CURVE_VARIANTS="${RANDOM_S_CURVE_VARIANTS:-12}"
export RANDOM_HARD_S_CURVE_VARIANTS="${RANDOM_HARD_S_CURVE_VARIANTS:-8}"
export RANDOM_ARC_VARIANTS="${RANDOM_ARC_VARIANTS:-8}"
export CORRIDOR_EPISODES_PER_COURSE="${CORRIDOR_EPISODES_PER_COURSE:-6}"
export CORRIDOR_LENGTH="${CORRIDOR_LENGTH:-28}"
export CORRIDOR_MAX_LATERAL_OFFSET_M="${CORRIDOR_MAX_LATERAL_OFFSET_M:-2.2}"
export REQUIRE_PHASES="${REQUIRE_PHASES:-launch,nominal,off_nominal,corridor}"
export PHASE_SAMPLING_WEIGHTS="${PHASE_SAMPLING_WEIGHTS:-launch=1.5,corridor=2.0,off_nominal=1.25}"

# Corridor and hard curves intentionally add stronger recovery labels. Keep the
# cap loose enough for the stress set while still catching broken labels.
export MAX_COMMAND_SATURATION_PCT="${MAX_COMMAND_SATURATION_PCT:-70}"

scripts/run_feature_policy_ablation.sh

#!/usr/bin/env bash
# Run the curve-stress GRU checkpoint closed-loop on harness curved courses.
#
# Generated hard_s_curve_rand_* courses are offline teacher curriculum only
# until matching simulator courses exist. This script uses the hardest current
# harness courses, circular_arc and s_curve, as the closed-loop DAgger bridge.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CS260C_LEARNED_CONTROLLER_CHECKPOINT="${CS260C_LEARNED_CONTROLLER_CHECKPOINT:-$REPO_ROOT/logs/learning_smoke/feature_bc_curve_stress_rejoin_no_prev_no_seq_leave_hard_s_curve_out_20e.npz}"
export ELODIN_LEARNED_COURSES="${ELODIN_LEARNED_COURSES:-circular_arc,s_curve}"
export ELODIN_LEARNED_CAMERA_PROFILES="${ELODIN_LEARNED_CAMERA_PROFILES:-vq1_pinhole}"
export ELODIN_LEARNED_OUT_DIR="${ELODIN_LEARNED_OUT_DIR:-$REPO_ROOT/logs/elodin_curve_stress_learned_suite}"
export ELODIN_LEARNED_SIM_TIME="${ELODIN_LEARNED_SIM_TIME:-18}"
export ELODIN_LEARNED_TIMEOUT_S="${ELODIN_LEARNED_TIMEOUT_S:-240}"
export ELODIN_LEARNED_IDLE_TIMEOUT_S="${ELODIN_LEARNED_IDLE_TIMEOUT_S:-60}"
export ELODIN_LEARNED_FRAME_STRIDE="${ELODIN_LEARNED_FRAME_STRIDE:-6}"

scripts/run_elodin_learned_suite.sh

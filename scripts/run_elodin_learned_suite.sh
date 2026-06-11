#!/usr/bin/env bash
# Run Elodin courses with the learned feature controller enabled.

set -euo pipefail

COURSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CS260C_LEARNED_CONTROLLER_CHECKPOINT="${CS260C_LEARNED_CONTROLLER_CHECKPOINT:-$COURSE_REPO/logs/learning_smoke/feature_bc_variants_leave_s_curve_out_20e_no_prev.npz}"
export CS260C_LEARNED_CONTROLLER_DEVICE="${CS260C_LEARNED_CONTROLLER_DEVICE:-cpu}"

if [[ ! -f "$CS260C_LEARNED_CONTROLLER_CHECKPOINT" ]]; then
  echo "Learned checkpoint not found: $CS260C_LEARNED_CONTROLLER_CHECKPOINT" >&2
  echo "Train it with learning.train_bc --no-prev-command-features, then export it with learning.export_policy_npz." >&2
  exit 1
fi

echo "[CS260C] learned checkpoint: $CS260C_LEARNED_CONTROLLER_CHECKPOINT"
echo "[CS260C] learned device: $CS260C_LEARNED_CONTROLLER_DEVICE"

"$COURSE_REPO/scripts/run_elodin_course_suite.py" \
  --courses "${ELODIN_LEARNED_COURSES:-easy,lateral_soft,low_high,four_gate_straight,circular_arc,s_curve}" \
  --out-dir "${ELODIN_LEARNED_OUT_DIR:-$COURSE_REPO/logs/elodin_learned_suite}" \
  --camera-profiles "${ELODIN_LEARNED_CAMERA_PROFILES:-${ELODIN_CAMERA_PROFILE:-vq1_pinhole}}" \
  --sim-time "${ELODIN_LEARNED_SIM_TIME:-18}" \
  --timeout-s "${ELODIN_LEARNED_TIMEOUT_S:-240}" \
  --idle-timeout-s "${ELODIN_LEARNED_IDLE_TIMEOUT_S:-60}" \
  --frame-stride "${ELODIN_LEARNED_FRAME_STRIDE:-6}"

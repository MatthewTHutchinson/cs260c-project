#!/usr/bin/env bash
set -euo pipefail

scripts/run_elodin_course_suite.py \
  --courses "${COURSES:-circular,s_curve}" \
  --out-dir "${OUT_DIR:-logs/elodin_challenge_suite}" \
  --sim-time "${SIM_TIME:-16}" \
  --timeout-s "${TIMEOUT_S:-360}" \
  --idle-timeout-s "${IDLE_TIMEOUT_S:-70}" \
  --frame-stride "${FRAME_STRIDE:-8}"

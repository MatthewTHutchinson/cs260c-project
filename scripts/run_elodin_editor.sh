#!/usr/bin/env bash
# Launch the sibling Elodin harness with the active CS260C pilot.

set -euo pipefail

COURSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELODIN_REPO="${ELODIN_REPO:-/Users/matthewhutchinson/dev/elodin-ai-grand-prix}"
TRACE_PATH="${TRACE_PATH:-$COURSE_REPO/logs/elodin_pilot_trace_editor.csv}"
FRAME_DIR="${FRAME_DIR:-$COURSE_REPO/logs/elodin_fpv_frames}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
LOG_PATH="${LOG_PATH:-$COURSE_REPO/logs/elodin_editor_stdout.log}"
CLEAR_EDITOR_LOGS="${CLEAR_EDITOR_LOGS:-1}"

if [[ ! -d "$ELODIN_REPO" ]]; then
  echo "Elodin repo not found: $ELODIN_REPO" >&2
  exit 1
fi

cd "$ELODIN_REPO"

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$(dirname "$TRACE_PATH")" "$FRAME_DIR" "$(dirname "$LOG_PATH")"

if [[ "$CLEAR_EDITOR_LOGS" == "1" || "$CLEAR_EDITOR_LOGS" == "true" ]]; then
  rm -f "$TRACE_PATH" "$LOG_PATH"
  rm -rf "$FRAME_DIR"
  mkdir -p "$FRAME_DIR"
fi

exec > >(tee "$LOG_PATH") 2>&1

echo "[CS260C] Elodin repo: $ELODIN_REPO"
echo "[CS260C] editor stdout log: $LOG_PATH"
echo "[CS260C] pilot trace: $TRACE_PATH"
echo "[CS260C] FPV frame dir: $FRAME_DIR"
echo "[CS260C] clear old editor logs: $CLEAR_EDITOR_LOGS"

RACE_SOLVER="${RACE_SOLVER:-solver.cs260c_pilot}" \
CS260C_PILOT_TRACE="$TRACE_PATH" \
CS260C_FRAME_DIR="$FRAME_DIR" \
CS260C_FRAME_STRIDE="$FRAME_STRIDE" \
ELODIN_ENABLE_FPV="${ELODIN_ENABLE_FPV:-1}" \
ELODIN_REAL_TIME="${ELODIN_REAL_TIME:-1}" \
ELODIN_INTERACTIVE="${ELODIN_INTERACTIVE:-1}" \
  elodin editor sim/main.py

#!/usr/bin/env bash
# Fast local smoke test for the sibling Elodin AI Grand Prix harness.
#
# This uses the harness's inline Betaflight mode instead of the normal
# `elodin run` s10 handoff. That keeps the no-FPV check deterministic enough
# for course-project bring-up on a local Mac.

set -euo pipefail

ELODIN_REPO="${ELODIN_REPO:-/Users/matthewhutchinson/dev/elodin-ai-grand-prix}"
LOG_PATH="${LOG_PATH:-$ELODIN_REPO/elodin_smoke.log}"
SMOKE_TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-180}"

cleanup() {
  pkill -f "elodin run sim/main.py" >/dev/null 2>&1 || true
  pkill -f "elodin render-server" >/dev/null 2>&1 || true
  pkill -f "betaflight_SITL" >/dev/null 2>&1 || true
}

if [[ ! -d "$ELODIN_REPO" ]]; then
  echo "Elodin repo not found: $ELODIN_REPO" >&2
  exit 1
fi

cleanup

cd "$ELODIN_REPO"

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

case "${ELODIN_ENABLE_FPV:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "run_elodin_smoke.sh is a no-FPV control smoke test." >&2
    echo "Use elodin editor sim/main.py in the Elodin repo for FPV/editor inspection." >&2
    exit 2
    ;;
esac

rm -f "$LOG_PATH"

ELODIN_SIM_TIME="${ELODIN_SIM_TIME:-2}" \
ELODIN_ENABLE_FPV="${ELODIN_ENABLE_FPV:-0}" \
ELODIN_REAL_TIME="${ELODIN_REAL_TIME:-0}" \
ELODIN_INTERACTIVE="${ELODIN_INTERACTIVE:-0}" \
ELODIN_MANAGE_BETAFLIGHT_INLINE=1 \
  uv run python sim/main.py run --no-s10 >"$LOG_PATH" 2>&1 &

pid=$!
deadline=$((SECONDS + SMOKE_TIMEOUT_SEC))

while kill -0 "$pid" >/dev/null 2>&1; do
  if grep -q "Simulation complete!" "$LOG_PATH" 2>/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for Elodin smoke run." >&2
    tail -80 "$LOG_PATH" >&2 || true
    cleanup
    exit 1
  fi
  sleep 1
done

tail -80 "$LOG_PATH"

if ! grep -q "SUCCESS: SITL integration working! Drone took off!" "$LOG_PATH"; then
  echo "Elodin smoke run did not report successful takeoff." >&2
  cleanup
  exit 1
fi

cleanup

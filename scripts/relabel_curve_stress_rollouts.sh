#!/usr/bin/env bash
# Relabel curve-stress learned rollouts and retrain the feature policy.
#
# Run scripts/run_curve_stress_closed_loop.sh first. This script consumes the
# resulting harness traces, creates privileged rejoin labels, then calls the
# existing source-aware relabel training wrapper with the curve-stress teacher
# dataset as the base.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_PROJECT_PYTHON="/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python"
if [[ -n "${PROJECT_PYTHON:-}" ]]; then
  PYTHON_BIN="$PROJECT_PYTHON"
elif [[ -x "$DEFAULT_PROJECT_PYTHON" ]]; then
  PYTHON_BIN="$DEFAULT_PROJECT_PYTHON"
else
  PYTHON_BIN="python"
fi

ROLLOUT_DIR="${ROLLOUT_DIR:-logs/elodin_curve_stress_learned_suite}"
CAMERA_PROFILES="${CAMERA_PROFILES:-vq1_pinhole}"
COURSES="${COURSES:-circular_arc,s_curve}"
RELABEL_OUT_DIR="${RELABEL_OUT_DIR:-logs/privileged_teacher/closed_loop_relabels/curve_stress}"
TRAIN_AFTER_RELABEL="${TRAIN_AFTER_RELABEL:-1}"
REQUIRE_TRACES="${REQUIRE_TRACES:-1}"
RUN_NAME="${RUN_NAME:-feature_bc_curve_stress_rejoin_plus_relabels_no_prev_no_seq_20e}"
RELABEL_ALLOWED_STATE_COMMAND_SOURCES="${RELABEL_ALLOWED_STATE_COMMAND_SOURCES:-learned}"
RELABEL_MAX_REFERENCE_ERROR_M="${RELABEL_MAX_REFERENCE_ERROR_M:-3.0}"
RELABEL_MAX_PAST_GATE_M="${RELABEL_MAX_PAST_GATE_M:-1.0}"

# hard_s_curve_rand_000 is the offline held-out hard curve from the curve-stress
# teacher dataset. We keep evaluating it during relabel training to catch
# regressions from closed-loop harness relabels.
export BASE_TRACE="${BASE_TRACE:-logs/privileged_teacher/trace_curve_stress_rejoin.csv}"
export COURSE="${COURSE:-hard_s_curve_rand_000}"
export MAX_COMMAND_SATURATION_PCT="${MAX_COMMAND_SATURATION_PCT:-70}"
export RUN_NAME

mkdir -p "$RELABEL_OUT_DIR"

IFS=',' read -r -a COURSE_ARRAY <<< "$COURSES"
IFS=',' read -r -a PROFILE_ARRAY <<< "$CAMERA_PROFILES"

RELABEL_PATHS=()
for profile in "${PROFILE_ARRAY[@]}"; do
  profile="${profile//[[:space:]]/}"
  if [[ -z "$profile" ]]; then
    continue
  fi
  for course in "${COURSE_ARRAY[@]}"; do
    course="${course//[[:space:]]/}"
    if [[ -z "$course" ]]; then
      continue
    fi
    trace="$ROLLOUT_DIR/$profile/$course/trace.csv"
    out="$RELABEL_OUT_DIR/${course}_${profile}_curve_stress_rejoin.csv"
    if [[ ! -f "$trace" ]]; then
      message="[CS260C] missing rollout trace=$trace"
      if [[ "$REQUIRE_TRACES" == "1" ]]; then
        echo "$message" >&2
        exit 1
      fi
      echo "$message; skipping" >&2
      continue
    fi

    "$PYTHON_BIN" scripts/relabel_closed_loop_trace.py \
      --trace "$trace" \
      --course "$course" \
      --teacher rejoin \
      --allowed-state-command-sources "$RELABEL_ALLOWED_STATE_COMMAND_SOURCES" \
      --max-reference-error-m "$RELABEL_MAX_REFERENCE_ERROR_M" \
      --max-past-gate-m "$RELABEL_MAX_PAST_GATE_M" \
      --episode-id "${course}:closed_loop:curve_stress_${profile}" \
      --out "$out"
    RELABEL_PATHS+=("$out")
  done
done

if [[ ${#RELABEL_PATHS[@]} -eq 0 ]]; then
  echo "[CS260C] no relabels were created" >&2
  exit 1
fi

printf '[CS260C] relabels=%s\n' "${RELABEL_PATHS[*]}"

if [[ "$TRAIN_AFTER_RELABEL" != "1" ]]; then
  exit 0
fi

export RELABELS="${RELABEL_PATHS[*]}"
scripts/run_closed_loop_relabel_training.sh

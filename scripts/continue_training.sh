#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/continue_training.log"

MULTI_CFG="configs/multimodal_obs_v2.yaml"
MULTI_BC="logs/bc_multimodal_obs_v2"
MULTI_DAG="logs/dagger_multimodal_obs_v2"
MULTI_PPO="logs/ppo_multimodal_obs_v2"
MULTI_BEST="$MULTI_PPO/policy_ppo_best.pt"

BIDIR_CFG="configs/generalization_bidirectional_obs_v1.yaml"
BIDIR_BC="logs/bc_generalization_bidirectional_obs_v1"
BIDIR_DAG="logs/dagger_generalization_bidirectional_obs_v1"
BIDIR_PPO="logs/ppo_generalization_bidirectional_obs_v1"
BIDIR_BEST="$BIDIR_PPO/policy_ppo_best.pt"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  local line="[$(timestamp)] $*"
  echo "$line"
  if [[ -n "${LOG_FILE:-}" ]]; then
    echo "$line" >> "$LOG_FILE"
  fi
}

wait_for_active_multimodal() {
  local pid
  pid="$(pgrep -f "train_all.py --config ${MULTI_CFG}" | head -n 1 || true)"
  if [[ -n "${pid:-}" ]]; then
    log "Detected active multimodal_obs_v2 run (PID $pid). Waiting for it to finish..."
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
    done
    log "multimodal_obs_v2 process PID $pid exited."
  else
    log "No active multimodal_obs_v2 process detected."
  fi
}

resume_multimodal_if_needed() {
  if [[ -f "$MULTI_BEST" ]]; then
    log "multimodal_obs_v2 already has PPO best checkpoint at $MULTI_BEST"
    return 0
  fi

  log "multimodal_obs_v2 is incomplete. Resuming pipeline with --resume."
  caffeinate -dimsu "$PYTHON" train_all.py \
    --config "$MULTI_CFG" \
    --bc-out "$MULTI_BC" \
    --dagger-out "$MULTI_DAG" \
    --ppo-out "$MULTI_PPO" \
    --resume
}

evaluate_multimodal_if_ready() {
  if [[ ! -f "$MULTI_BEST" ]]; then
    log "Skipping multimodal evaluation because $MULTI_BEST is still missing."
    return 0
  fi

  log "Evaluating multimodal_obs_v2 best checkpoint on standard multimodal suite."
  "$PYTHON" -m eval.evaluate_track_suite \
    --config "$MULTI_CFG" \
    --type ppo \
    --ckpt "$MULTI_BEST" \
    --episodes 8 \
    | tee "$MULTI_PPO/eval_standard_suite.txt"

  log "Evaluating multimodal_obs_v2 best checkpoint on competition-spec multimodal suite."
  "$PYTHON" -m eval.evaluate_track_suite \
    --config configs/competition_spec_multimodal_eval.yaml \
    --type ppo \
    --ckpt "$MULTI_BEST" \
    --episodes 8 \
    | tee "$MULTI_PPO/eval_competition_spec_suite.txt"

  log "Evaluating multimodal_obs_v2 best checkpoint on extended directional/long-course multimodal suite."
  "$PYTHON" -m eval.evaluate_track_suite \
    --config configs/extended_generalization_multimodal_eval.yaml \
    --type ppo \
    --ckpt "$MULTI_BEST" \
    --episodes 8 \
    | tee "$MULTI_PPO/eval_extended_generalization_suite.txt"
}

launch_bidirectional_branch() {
  if pgrep -f "train_all.py --config ${BIDIR_CFG}" >/dev/null 2>&1; then
    log "Bidirectional branch is already running."
    return 0
  fi

  if [[ -f "$BIDIR_BEST" ]]; then
    log "Bidirectional branch already has a PPO best checkpoint at $BIDIR_BEST"
    return 0
  fi

  log "Launching bidirectional/long-course state branch."
  caffeinate -dimsu "$PYTHON" train_all.py \
    --config "$BIDIR_CFG" \
    --bc-out "$BIDIR_BC" \
    --dagger-out "$BIDIR_DAG" \
    --ppo-out "$BIDIR_PPO" \
    --resume
}

evaluate_bidirectional_if_ready() {
  if [[ ! -f "$BIDIR_BEST" ]]; then
    log "Skipping bidirectional evaluation because $BIDIR_BEST is still missing."
    return 0
  fi

  log "Evaluating bidirectional branch on the extended directional/long-course suite."
  "$PYTHON" -m eval.evaluate_track_suite \
    --config configs/extended_generalization_obs_eval.yaml \
    --type ppo \
    --ckpt "$BIDIR_BEST" \
    --episodes 8 \
    | tee "$BIDIR_PPO/eval_extended_generalization_suite.txt"

  log "Evaluating bidirectional branch on the legacy richer-state suite."
  "$PYTHON" -m eval.evaluate_track_suite \
    --config configs/generalization_obs_v1.yaml \
    --type ppo \
    --ckpt "$BIDIR_BEST" \
    --episodes 8 \
    | tee "$BIDIR_PPO/eval_legacy_suite.txt"
}

main() {
  log "continue_training.sh starting"
  wait_for_active_multimodal
  resume_multimodal_if_needed
  evaluate_multimodal_if_ready
  launch_bidirectional_branch
  evaluate_bidirectional_if_ready
  log "continue_training.sh finished"
}

main "$@"

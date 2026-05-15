#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

print_section() {
  echo
  echo "== $1 =="
}

latest_file() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -print0 2>/dev/null \
    | xargs -0 ls -t 2>/dev/null \
    | head -n 1
}

summarize_dir() {
  local label="$1"
  local dir="$2"
  local latest
  latest="$(latest_file "$dir")"
  if [[ -z "${latest:-}" ]]; then
    echo "$label: no files yet ($dir)"
    return 0
  fi
  local stamp
  stamp="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$latest" 2>/dev/null || true)"
  echo "$label: $latest"
  if [[ -n "${stamp:-}" ]]; then
    echo "  updated: $stamp"
  fi
}

print_section "Active Processes"
ps -Ao pid,etime,%cpu,%mem,command | rg 'multimodal_obs_v2|generalization_bidirectional_obs_v1|continue_training.sh|training\.(bc|dagger|ppo)|train_all.py' || true

print_section "Checkpoint Directories"
summarize_dir "multimodal v2 BC" "logs/bc_multimodal_obs_v2"
summarize_dir "multimodal v2 DAgger" "logs/dagger_multimodal_obs_v2"
summarize_dir "multimodal v2 PPO" "logs/ppo_multimodal_obs_v2"
summarize_dir "bidirectional BC" "logs/bc_generalization_bidirectional_obs_v1"
summarize_dir "bidirectional DAgger" "logs/dagger_generalization_bidirectional_obs_v1"
summarize_dir "bidirectional PPO" "logs/ppo_generalization_bidirectional_obs_v1"

print_section "Continuation Log"
if [[ -f logs/continue_training.log ]]; then
  tail -n 20 logs/continue_training.log
else
  echo "logs/continue_training.log not found"
fi

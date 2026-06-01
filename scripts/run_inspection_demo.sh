#!/usr/bin/env bash
# Generate the current synthetic gate-inspection assets with the known Python.

set -euo pipefail

COURSE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PYTHON="${PROJECT_PYTHON:-/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python}"
OUT_DIR="${OUT_DIR:-$COURSE_REPO/logs/gate_inspection_sequence}"

if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "Project Python not found: $PROJECT_PYTHON" >&2
  echo "Set PROJECT_PYTHON=/path/to/python and rerun." >&2
  exit 1
fi

cd "$COURSE_REPO"

"$PROJECT_PYTHON" scripts/inspect_gate_frames.py \
  --demo \
  --demo-frames 42 \
  --out-dir "$OUT_DIR" \
  --save-mask

"$PROJECT_PYTHON" scripts/plot_pilot_trace.py \
  --trace "$OUT_DIR/trace.csv" \
  --out "$OUT_DIR/pilot_trace.png" \
  --title "Synthetic Gate Approach: Detection to Control"

echo "Wrote: $OUT_DIR"

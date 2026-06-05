# Feature Policy Learning

This package is the first learning scaffold for the active FPV racing stack.
It is intentionally feature-based:

```text
classical CV gate features + telemetry + tracker history
  -> GRU/MLP policy
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

It does not train a raw-image CNN yet, and it does not use global pose,
simulator gate IDs, or known gate coordinates as policy inputs.

## Important Caveat

The current local trace CSVs prove the training plumbing works. They should not
be treated as high-quality racing demonstrations.

Known teacher problems:

- the visual-servo controller can bias toward gate edges/corners when detections
  are clipped or off-center
- lateral displacement and curved-track turns are weak
- circular and S-shaped courses expose the missing lookahead/navigation layer

Behavioral cloning will copy those mistakes. Use existing traces for smoke
tests and debugging only until better teacher labels are generated.

## T4 Setup

From the repo root on the GCP VM:

```bash
conda activate cs260c-t4
python -m pip install -r requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Smoke Test

Overfit a synthetic trace:

```bash
python -m learning.train_bc \
  --demo-synthetic \
  --epochs 20 \
  --batch-size 64 \
  --out checkpoints/feature_bc_synthetic.pt
```

Evaluate the checkpoint:

```bash
python -m learning.eval_policy \
  --checkpoint checkpoints/feature_bc_synthetic.pt \
  --traces logs/learning_synthetic/trace.csv
```

## Local Trace Smoke Training

Point the same trainer at trace CSVs only to verify the loader/trainer on real
columns:

```bash
python -m learning.train_bc \
  --traces logs/elodin_course_suite \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_local_traces.pt
```

The loader recursively finds `trace.csv` files under directories.

Do not present this checkpoint as a capable autonomy policy. It is a plumbing
checkpoint.

## Data We Still Want

For stronger BC/DAgger training, future trace rows should include:

- detector/tracker features: bearing, range, confidence, mode, age
- telemetry: attitude, angular rates, linear velocity, acceleration when available
- previous command
- teacher command or local target command
- run metadata: course, camera profile, pass/fail status

The useful next labels should come from a better teacher:

- sequence-aware local corridor target
- minimum-snap or smooth lookahead reference for known debug courses
- improved hand-controller that targets the gate centerline and handles lateral
  displacement before we clone it

## Privileged Teacher Dataset

Generate a first privileged-teacher dataset from known debug course geometry:

```bash
python scripts/generate_privileged_teacher_dataset.py \
  --out logs/privileged_teacher/trace.csv
```

Audit the teacher before training on it:

```bash
python scripts/audit_privileged_teacher_dataset.py \
  --trace logs/privileged_teacher/trace.csv \
  --plot \
  --out-dir logs/privileged_teacher/audit
```

Train on it:

```bash
python -m learning.train_bc \
  --traces logs/privileged_teacher/trace.csv \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_privileged_teacher.pt
```

The loader prefers `teacher_*` target columns when they exist. It does not use
`world_*` or `teacher_next_gate_*` columns as student inputs.

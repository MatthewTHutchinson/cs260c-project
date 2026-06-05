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

## Real Trace Training

Point the same trainer at trace CSVs:

```bash
python -m learning.train_bc \
  --traces logs/elodin_course_suite \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_local_traces.pt
```

The loader recursively finds `trace.csv` files under directories.

## Data We Still Want

For stronger BC/DAgger training, future trace rows should include:

- detector/tracker features: bearing, range, confidence, mode, age
- telemetry: attitude, angular rates, linear velocity, acceleration when available
- previous command
- teacher command or local target command
- run metadata: course, camera profile, pass/fail status


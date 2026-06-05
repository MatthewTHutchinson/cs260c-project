# GCP T4 Handoff

Date: 2026-06-04

Purpose: use the GCP T4 machine for compact training experiments, not for
official Windows-simulator bring-up.

Current machine snapshot from diagnostics:

- Ubuntu 22.04.5 LTS
- 4 vCPU
- 14 GiB RAM
- ~146 GiB free disk on `/`
- Tesla T4 with 15360 MiB VRAM
- NVIDIA driver 580.126.20
- CUDA 12.9 toolkit visible via `nvcc`
- base Python is 3.13.13
- PyTorch 2.12.0+cu130 sees the GPU successfully

Near-term learning direction:

```text
classical CV gate features + telemetry + tracker history
  -> GRU/MLP temporal policy
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

Do not start by training a raw-image CNN. Use an open-source gate detector later
if it can be plugged in cleanly, or train/evaluate a detector from open datasets
after the feature-policy baseline exists.

## 1. T4 Machine Diagnostics

Run this on the GCP VM and save the output:

```bash
cat > /tmp/cs260c_t4_diagnostics.sh <<'SH'
set -euxo pipefail

echo "=== OS ==="
uname -a
cat /etc/os-release || true

echo "=== CPU/RAM/DISK ==="
lscpu | sed -n '1,25p' || true
free -h || true
df -h / || true

echo "=== GPU ==="
which nvidia-smi || true
nvidia-smi || true
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap,power.limit --format=csv || true

echo "=== CUDA/NVCC ==="
which nvcc || true
nvcc --version || true
ls -la /usr/local | grep cuda || true

echo "=== Python ==="
which python || true
python --version || true
which python3 || true
python3 --version || true
which pip || true
pip --version || true

echo "=== Conda/Mamba/Uv ==="
which conda || true
conda --version || true
which mamba || true
mamba --version || true
which uv || true
uv --version || true

echo "=== Git ==="
which git || true
git --version || true

echo "=== PyTorch Probe ==="
python3 - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_version", torch.version.cuda)
    print("gpu_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu_name", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_probe_error", repr(exc))
PY
SH

bash /tmp/cs260c_t4_diagnostics.sh | tee ~/cs260c_t4_diagnostics.txt
```

Send or paste the contents of:

```bash
~/cs260c_t4_diagnostics.txt
```

The key things to check are:

- T4 is visible in `nvidia-smi`
- available VRAM is close to 15 GB
- Python version is 3.10 or 3.11
- PyTorch either already sees CUDA or can be installed cleanly
- enough disk is available for logs/checkpoints

## 2. Clone The Repo

Preferred location:

```bash
mkdir -p ~/dev
cd ~/dev
git clone https://github.com/MatthewTHutchinson/cs260c-project.git
cd cs260c-project
```

If the latest local work has not been pushed yet, either push it from the Mac or
copy this document manually onto the VM.

Basic repo check:

```bash
git status --short --branch
git log --oneline -5
```

## 3. Python Environment

Use a separate conda env or venv for the project. Do not rely on the base
Python 3.13 environment for the training stack unless every dependency is
verified against it.

Recommended choice: a fresh Python 3.11 environment.

Using conda:

```bash
conda create -n cs260c-t4 python=3.11 -y
conda activate cs260c-t4
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

Using venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

Then install PyTorch for CUDA. First try the official CUDA 12.1 wheels:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If CUDA is not available, stop and inspect the diagnostics before spending more
cloud time.

If Python 3.11 is not available, install it first or use conda rather than the
base 3.13 interpreter.

## 4. Local Repo Smoke Tests On T4

These should run without simulator access:

```bash
conda activate cs260c-t4
python -m py_compile algorithm/*.py scripts/*.py
python -m py_compile learning/*.py
python scripts/audit_sign_conventions.py
python scripts/audit_sequence_selection.py
python scripts/audit_lateral_reacquisition.py
python scripts/inspect_gate_frames.py --demo --demo-frames 3 --out-dir logs/t4_demo_inspection --save-mask
```

Expected: no Python errors, and demo inspection writes overlays/masks.

## 5. First Training Work

The first training scaffold now exists under:

```bash
ls learning
sed -n '1,220p' learning/README.md
```

The first model input should be low-dimensional:

```text
gate bearing
gate range estimate
gate confidence
tracker mode one-hot
recent bearing/range deltas
attitude/orientation
angular rates
linear velocity if available
previous command
```

The first model output should match the current command boundary:

```text
roll_rate
pitch_rate
yaw_rate
thrust
```

Run the synthetic smoke test first:

```bash
python -m learning.train_bc \
  --demo-synthetic \
  --epochs 20 \
  --batch-size 64 \
  --out checkpoints/feature_bc_synthetic.pt

python -m learning.eval_policy \
  --checkpoint checkpoints/feature_bc_synthetic.pt \
  --traces logs/learning_synthetic/trace.csv
```

Then try local trace data if available:

```bash
python -m learning.train_bc \
  --traces logs/elodin_course_suite \
  --epochs 40 \
  --batch-size 128 \
  --out checkpoints/feature_bc_local_traces.pt
```

## 6. First Experiment

Goal: prove the model and data plumbing work, not produce a champion racer.

Experiment 0:

```text
tiny dataset -> overfit one log -> loss goes near zero
```

Experiment 1:

```text
train on simple-track logs
validate on held-out simple-track logs
report command MSE and mode-conditioned errors
```

Experiment 2:

```text
add curved-track logs
compare current reactive teacher vs lookahead/minimum-snap teacher labels
```

Only after these work should PPO or raw-image CNNs be considered.

## 7. Open-Source Gate CNN Position

Using an open-source gate CNN is a good idea if it is genuinely drop-in:

- accepts normal RGB frames or easily adapted camera intrinsics
- outputs gate center/corners/box/range in a usable format
- can run in real time on CPU or modest GPU
- has weights available, not just training code
- license allows use
- can be wrapped behind `algorithm/neural_gate_detector.py`

If any of those are missing, defer it. Classical CV features are enough for the
first learned-control experiment.

## 8. Cloud Budget Rule

Before starting a long job:

```bash
nvidia-smi
git status --short
python -m py_compile algorithm/*.py scripts/*.py learning/*.py
```

Run a 1-2 minute smoke test first. Then run the longer job.

Stop the VM when done.

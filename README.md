# CS 260C Project: Autonomous Drone Gate Racing

This repo is now organized around the current VQ1/final-project algorithm:

```text
FPV gate recognition
  -> temporal tracking
  -> reactive navigation
  -> body-rate/thrust or RC command adapter
```

The old PyBullet / BC / DAgger / PPO stack has been quarantined in:

```text
legacy/pybullet/
```

It is preserved for reference, but it is not the active project direction.

## Active Map

- `algorithm/`: current autonomous racing algorithm and adapter helpers.
- `docs/`: current project docs, competition context, and final-project framing.
- `scripts/`: current operational scripts.
- `docs/reference/`: official specs, MAVLink schema, papers, and external reference captures.
- `external/`: ignored local research checkouts such as GateNet.
- `legacy/pybullet/`: old simulator, tracks, policies, configs, training code, and presentation assets.

## Start Here

Read these in order:

1. `docs/AUTONOMOUS_RACING_ALGORITHM.md`
2. `docs/ELODIN_PRACTICE_HARNESS.md`
3. `docs/FINAL_PRESENTATION_DECK.md`
4. `docs/COMPETITION_CONTEXT.md`
5. `docs/VQ1_ACCESS_RUNBOOK.md`

## Elodin Harness

The local practice harness lives outside this repo:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix
```

Current status:

- Elodin CLI installed.
- Elodin DB CLI installed.
- `uv sync` completed.
- Elodin tests passed.
- Betaflight SITL built.
- The no-FPV Betaflight/control smoke path is reliable through the local wrapper.
- The CS260C solver adapter has been smoke-tested in Elodin without consuming world pose.

No-FPV smoke wrapper:

```bash
scripts/run_elodin_smoke.sh
RACE_SOLVER=solver.cs260c_pilot scripts/run_elodin_smoke.sh
```

Use the editor wrapper for FPV inspection and command tracing:

```bash
scripts/run_elodin_editor.sh
```

The smoke wrapper intentionally avoids camera rendering.

Inspect detector/controller behavior on images, videos, or a synthetic gate demo:

```bash
scripts/run_inspection_demo.sh
scripts/inspect_gate_frames.py --demo --out-dir logs/gate_inspection_demo --save-mask
scripts/inspect_gate_frames.py --demo --demo-frames 42 --out-dir logs/gate_inspection_sequence --save-mask
scripts/plot_pilot_trace.py \
  --trace logs/gate_inspection_sequence/trace.csv \
  --out logs/gate_inspection_sequence/pilot_trace.png
scripts/inspect_gate_frames.py --source path/to/frames_or_video --out-dir logs/gate_inspection_real
```

Swap in a GateNet-style ONNX export for detector-only validation:

```bash
CS260C_GATE_DETECTOR=gatenet \
CS260C_GATE_DETECTOR_MODEL=models/gatenet.onnx \
CS260C_GATE_DETECTOR_OUTPUT=corners8 \
scripts/inspect_gate_frames.py \
  --source logs/elodin_fpv_frames \
  --out-dir logs/gatenet_inspection_elodin
```

The repo does not include GateNet weights. Treat this as an adapter path until
an exported model is available and its overlays beat the classical baseline.
The upstream reference clone lives at `external/gatenet/`; see
`docs/reference/gatenet_external.md`.

If direct script commands pick the wrong Python, use:

```bash
PROJECT_PYTHON=/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/run_inspection_demo.sh
```

## Python

The active algorithm package has a small dependency surface:

```bash
python -m pip install -r requirements.txt
```

On this machine, the known-good project interpreter is:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python
```

Validate active code:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m py_compile algorithm/*.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sequence_selection.py
```

## Project Rule

Do not mix new VQ1/Elodin work into `legacy/pybullet/`.

Do not present old PyBullet track results as the final algorithm. The final project should present the current autonomous racing algorithm and use PyBullet only as historical context for why the simulator strategy changed.

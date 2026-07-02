# Project Archive And Restart

Date archived: 2026-07-01

This repository is the durable backup for the CS 260C autonomous drone racing
project. The large generated frame dumps and simulator build directories are
not stored in GitHub, but the source code, documentation, final report,
reproducible Elodin patch, final teacher dataset, final learned-policy
checkpoints, and key result traces are stored here.

## What Is Preserved

- `algorithm/`: competition-facing perception, tracking, control, and adapters.
- `learning/`: GRU/MLP behavior-cloning implementation and evaluation code.
- `scripts/`: teacher generation, audits, training, simulator wrappers, and plots.
- `docs/`: architecture, competition context, decisions, audits, and runbooks.
- `overleaf_final/`: final report source and figures.
- `patches/elodin-ai-grand-prix-cs260c.patch`: all local changes to the external
  Elodin practice harness, including the CS260C solver.
- `patches/gym-pybullet-drones-local.patch.gz`: the remaining local changes from
  the historical PyBullet checkout, preserved before that checkout was deleted.
- `environment.yml`: snapshot of the known-good macOS Conda environment.
- `artifacts/final/`: compact final dataset, checkpoints, and representative
  closed-loop traces used by the report.

The following were intentionally not archived:

- raw FPV frame dumps under `logs/`
- repeated intermediate checkpoints and generated plots
- Elodin build products and databases
- the built Betaflight submodule
- ignored external research checkouts such as GateNet

Those files consumed most of the local disk space and can be regenerated.

## Clone And Recreate The Python Environment

```bash
mkdir -p ~/dev
cd ~/dev
git clone https://github.com/MatthewTHutchinson/cs260c-project.git
cd cs260c-project

conda env create -f environment.yml
conda activate cs260c-project
```

For a lighter environment, create Python 3.10 or 3.11 and install
`requirements.txt`, then install PyTorch separately.

## Restore The Elodin Practice Harness

```bash
cd ~/dev
git clone --recurse-submodules https://github.com/elodin-sys/ai-grand-prix.git elodin-ai-grand-prix
cd elodin-ai-grand-prix
git apply ../cs260c-project/patches/elodin-ai-grand-prix-cs260c.patch
uv sync
```

Rebuild Betaflight using the project wrapper:

```bash
cd ~/dev/cs260c-project
scripts/build_betaflight.sh
```

The patch was compared against the local Elodin checkout at archive time and
matched exactly. Betaflight build artifacts are not part of the patch.

## Restore Final Artifacts To Their Runtime Paths

The archived artifacts can be used directly, or copied back to the historical
`logs/` paths expected by older commands:

```bash
mkdir -p logs/privileged_teacher logs/learning_smoke
cp artifacts/final/teacher_curve_stress_rejoin_corridor.csv \
  logs/privileged_teacher/trace_curve_stress_rejoin_corridor.csv
cp artifacts/final/feature_policy_curve_stress_corridor.pt \
  logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e.pt
cp artifacts/final/feature_policy_curve_stress_corridor.npz \
  logs/learning_smoke/feature_bc_curve_stress_rejoin_corridor_no_prev_no_seq_leave_hard_s_curve_out_20e.npz
```

## Validate The Restored Project

```bash
python -m py_compile algorithm/*.py learning/*.py scripts/*.py
python scripts/audit_course_alignment.py
python scripts/audit_sign_conventions.py
python scripts/audit_sequence_selection.py
python scripts/audit_learning_feature_spec.py \
  --no-prev-command-features \
  --no-sequence-features \
  --expect-no-prev-command-features \
  --expect-no-sequence-features
```

Then run the local simulator checks:

```bash
scripts/run_elodin_smoke.sh
scripts/run_elodin_course_suite.py \
  --courses easy,lateral_soft,low_high,four_gate_straight
```

## Where To Resume

Start with:

1. `docs/PROJECT_DIRECTION_AUDIT_2026-06-05.md`
2. `docs/DUE_TONIGHT_ROADMAP_STATUS_2026-06-10.md`
3. `docs/NEXT_STEPS_LEARNED_POLICY.md`
4. `overleaf_final/main.tex`

The unresolved technical issue is closed-loop learned control on curved tracks.
Offline behavior cloning fit improved, but the safety supervisor often selected
`reactive_fallback`, so the student policy did not receive consistent command
authority. The next experiment should audit learned-versus-fallback command
ownership and Betaflight/MAVSDK command scaling before PPO.

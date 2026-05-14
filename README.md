# CS 260C Project: Autonomous Drone Gate Racing

This repository contains a staged imitation-learning and reinforcement-learning pipeline for autonomous quadrotor gate racing in `gym-pybullet-drones`.

The current training flow is:

1. Behavior Cloning (BC)
2. DAgger dataset aggregation
3. PPO fine-tuning from the DAgger checkpoint

## Current best checkpoints

- Best overall current model:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
  with richer-observation hard-suite return `82.49`
  and richer-observation easy-suite return `82.95`

- New vision branch status:
  onboard RGB rendering, detector-backed `vision_bridge` observations,
  scene visual randomization, and multimodal state+vision training are now implemented.
  The first multimodal config is `configs/multimodal_obs_v1.yaml`.
  A teacher-warmstarted follow-up is prepared in `configs/multimodal_obs_v2.yaml`.

If you want the strongest current model, start with `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`.

## Competition context

For external AI Grand Prix / DCL-facing facts, use:

- `docs/COMPETITION_NOTES.md`
- backing PDF: `docs/260508_Technical_Spec_0002.pdf`

The older Gmail PDF exports in `docs/` are still useful context, but they should be treated as historical and possibly stale unless the latest spec also confirms them.

## Project map

- `env/`: drone racing environment, gate logic, and future DCL adapter
- `expert/`: expert policy used to generate imitation labels
- `policy/`: BC actor and PPO actor-critic networks
- `training/`: BC, DAgger, and PPO training scripts
- `eval/`: evaluation, plotting, and simulator visualization
- `configs/default.yaml`: default hyperparameters and environment settings
- `docs/`: progress tracking, report notes, and portfolio notes
- `CLAUDE.md`: working context and recent change log for Claude Code

## Environment

This repo expects:

- a Python environment with packages from `requirements.txt`
- `gym-pybullet-drones` installed editable from `../gym-pybullet-drones`

If your shell `python3` is missing dependencies, use:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python
```

## Common commands

Visualize the expert:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize --type expert
```

Run the full training pipeline:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py
```

Run PPO only from a DAgger checkpoint:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo \
  --config configs/default.yaml \
  --dagger-ckpt logs/dagger/policy_dagger.pt \
  --dagger-data logs/dagger \
  --out logs/ppo
```

Evaluate a trained PPO policy:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate \
  --type ppo \
  --ckpt logs/ppo/policy_ppo.pt \
  --episodes 20
```

Run multitrack PPO with held-out validation:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo \
  --config configs/multitrack_ppo.yaml \
  --dagger-ckpt logs/dagger/policy_dagger.pt \
  --dagger-data logs/dagger \
  --out logs/ppo_multitrack_v1
```

Evaluate a policy across the held-out track suite:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate_track_suite \
  --config configs/multitrack_ppo.yaml \
  --type ppo \
  --ckpt logs/ppo_multitrack_v1/policy_ppo_best.pt
```

Run the harder generalization pipeline with more tracks and randomized starts:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/generalization_hard.yaml \
  --bc-out logs/bc_generalization_v1 \
  --dagger-out logs/dagger_generalization_v1 \
  --ppo-out logs/ppo_generalization_v1
```

Evaluate the current harder-suite champion:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate_track_suite \
  --config configs/generalization_hard.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_v1/policy_ppo_best.pt
```

Run PPO with balanced checkpoint selection across both held-out suites:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo \
  --config configs/generalization_balanced.yaml \
  --dagger-ckpt logs/dagger_generalization_v1/policy_dagger.pt \
  --dagger-data logs/dagger_generalization_v1 \
  --out logs/ppo_generalization_balanced_v1
```

Run the zigzag-focused balanced PPO follow-up:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo \
  --config configs/generalization_balanced_v2.yaml \
  --dagger-ckpt logs/dagger_generalization_v1/policy_dagger.pt \
  --dagger-data logs/dagger_generalization_v1 \
  --out logs/ppo_generalization_balanced_v2
```

Train the richer-observation / stronger-expert branch:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/generalization_obs_v1.yaml \
  --bc-out logs/bc_generalization_obs_v1 \
  --dagger-out logs/dagger_generalization_obs_v1 \
  --ppo-out logs/ppo_generalization_obs_v1
```

Train the robustness-focused richer-observation branch:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/generalization_robust_obs_v1.yaml \
  --bc-out logs/bc_generalization_robust_obs_v1 \
  --dagger-out logs/dagger_generalization_robust_obs_v1 \
  --ppo-out logs/ppo_generalization_robust_obs_v1
```

Train the drop/recover-focused robustness follow-up:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/generalization_robust_obs_v2.yaml \
  --bc-out logs/bc_generalization_robust_obs_v2 \
  --dagger-out logs/dagger_generalization_robust_obs_v2 \
  --ppo-out logs/ppo_generalization_robust_obs_v2
```

Run the robustness audit on the current richer-observation champion:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.robustness_audit \
  --config configs/robustness_obs_v1.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt \
  --out logs/robustness_obs_v1
```

That audit writes:

- `logs/robustness_obs_v1/report.md`
- `logs/robustness_obs_v1/per_track.csv`
- `logs/robustness_obs_v1/summary.json`

Evaluate an existing state policy through the detector-backed perception bridge:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate \
  --config configs/vision_bridge_eval_v1.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt \
  --episodes 10
```

Visualize the same bridge path in PyBullet:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize \
  --config configs/vision_bridge_eval_v1.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt
```

Run a spec-aligned bridge evaluation with the latest competition camera assumptions:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate_track_suite \
  --config configs/competition_spec_bridge_eval.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt
```

Evaluate a multimodal checkpoint under spec-sized rendering with downsampled policy inputs:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate_track_suite \
  --config configs/competition_spec_multimodal_eval.yaml \
  --type ppo \
  --ckpt logs/ppo_multimodal_obs_v1/policy_ppo_best.pt
```

Train the first multimodal state+vision branch:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/multimodal_obs_v1.yaml \
  --bc-out logs/bc_multimodal_obs_v1 \
  --dagger-out logs/dagger_multimodal_obs_v1 \
  --ppo-out logs/ppo_multimodal_obs_v1
```

Train the teacher-warmstarted multimodal follow-up:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/multimodal_obs_v2.yaml \
  --bc-out logs/bc_multimodal_obs_v2 \
  --dagger-out logs/dagger_multimodal_obs_v2 \
  --ppo-out logs/ppo_multimodal_obs_v2
```

Resume an interrupted run from the latest completed stage and saved trainer state:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/multimodal_obs_v2.yaml \
  --bc-out logs/bc_multimodal_obs_v2 \
  --dagger-out logs/dagger_multimodal_obs_v2 \
  --ppo-out logs/ppo_multimodal_obs_v2 \
  --resume
```

On macOS, keep the machine awake during long training runs:

```bash
caffeinate -dimsu -w <train_all_pid>
```

This prevents idle sleep, but it will not survive shutdown, reboot, or closing the laptop lid.

Quick visualization presets:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize --preset state_champion
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize --preset multimodal_v1
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize --preset competition_multimodal_v1
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize --preset multimodal_v2_bc
```

Inspect the onboard camera input and detector overlay side by side:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize_perception --preset multimodal_v1
```

Save debug frames without opening an OpenCV window:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize_perception \
  --preset multimodal_v1 \
  --no-window \
  --save-dir logs/perception_debug_multimodal_v1
```

Run the extended directional / longer-course audit on the current state champion:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.evaluate_track_suite \
  --config configs/extended_generalization_obs_eval.yaml \
  --type ppo \
  --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt
```

Prepared next state-training branch for mirrored/right-turn plus longer-course generalization:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py \
  --config configs/generalization_bidirectional_obs_v1.yaml \
  --bc-out logs/bc_generalization_bidirectional_obs_v1 \
  --dagger-out logs/dagger_generalization_bidirectional_obs_v1 \
  --ppo-out logs/ppo_generalization_bidirectional_obs_v1
```

## Documentation workflow

Use these files as the shared documentation backbone:

- `docs/PROGRESS.md`: chronological experiment log and current status
- `docs/REPORT_NOTES.md`: class report outline, claims, and evidence
- `docs/PORTFOLIO_NOTES.md`: public-facing project story and visuals checklist
- `docs/BRAINSTORMING.md`: roadmap, limitations, and future observation / vision plans
- `docs/COMPETITION_NOTES.md`: current external source-of-truth notes, confidence labels, and open questions

The intended workflow is simple:

1. After each meaningful run, update `docs/PROGRESS.md`.
2. When a result matters for the report, summarize it in `docs/REPORT_NOTES.md`.
3. When a result is portfolio-worthy, translate it into the cleaner narrative in `docs/PORTFOLIO_NOTES.md`.
4. When a competition-facing assumption changes, reconcile it in `docs/COMPETITION_NOTES.md`.

Older drafts in the retired `brainstorming/` folder should no longer be treated as current documentation.

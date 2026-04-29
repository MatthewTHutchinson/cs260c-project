# CS 260C Project: Autonomous Drone Gate Racing

This repository contains a staged imitation-learning and reinforcement-learning pipeline for autonomous quadrotor gate racing in `gym-pybullet-drones`.

The current training flow is:

1. Behavior Cloning (BC)
2. DAgger dataset aggregation
3. PPO fine-tuning from the DAgger checkpoint

## Current best checkpoints

- Best easier-suite specialist:
  `logs/ppo_multitrack_v1/policy_ppo_best.pt`
  with held-out return `70.33` on `configs/multitrack_ppo.yaml`
- Best harder-suite generalization model:
  `logs/ppo_generalization_v1/policy_ppo_best.pt`
  with held-out return `71.68` on `configs/generalization_hard.yaml`

If you want the strongest all-around current model, start with `logs/ppo_generalization_v1/policy_ppo_best.pt`.

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

## Documentation workflow

Use these files as the shared documentation backbone:

- `docs/PROGRESS.md`: chronological experiment log and current status
- `docs/REPORT_NOTES.md`: class report outline, claims, and evidence
- `docs/PORTFOLIO_NOTES.md`: public-facing project story and visuals checklist

The intended workflow is simple:

1. After each meaningful run, update `docs/PROGRESS.md`.
2. When a result matters for the report, summarize it in `docs/REPORT_NOTES.md`.
3. When a result is portfolio-worthy, translate it into the cleaner narrative in `docs/PORTFOLIO_NOTES.md`.

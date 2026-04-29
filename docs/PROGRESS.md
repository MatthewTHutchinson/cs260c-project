# Progress Log

## Current snapshot

Date: 2026-04-29

Current status:

- The current strongest generalization checkpoint is `logs/ppo_generalization_v1/policy_ppo_best.pt`.
- On the harder held-out suite from `configs/generalization_hard.yaml`, that checkpoint achieved aggregate completion `100%`, crash `0%`, mean gates `4.18`, and mean return `71.68`.
- On the earlier held-out suite from `configs/multitrack_ppo.yaml`, the same checkpoint still achieved aggregate completion `100%`, crash `0%`, mean gates `4.00`, and mean return `69.57`.
- The older `logs/ppo_multitrack_v1/policy_ppo_best.pt` remains slightly better on the easier held-out suite with mean return `70.33`, so there is now a clear robustness-versus-specialization tradeoff rather than a broken pipeline.
- A new mixed-validation PPO run is now in progress at `logs/ppo_generalization_balanced_v1`, using both the easier and harder held-out suites for checkpoint selection.

## Immediate next step

Let `logs/ppo_generalization_balanced_v1` finish, then compare its best checkpoint against both `ppo_generalization_v1` and `ppo_multitrack_v1` on both validation suites.

Suggested direction:

- judge checkpoints with a combined validation score instead of only one distribution
- check whether the balanced run narrows the easier-suite gap while keeping the harder-suite advantage
- if the tradeoff persists, consider weighted validation or curriculum adjustments instead of only more training

## What to record after each run

- run name
- config changes
- training duration
- whether the policy flew, hovered, oscillated, or crashed
- completion rate
- crash rate
- mean gates passed
- best checkpoint if intermediate checkpoints looked better than final

## Experiment log template

### Run: `ppo_stable_v1`

- Date: 2026-04-28
- Goal: Stabilize PPO enough to preserve BC flight quality while improving return.
- Starting checkpoint: `logs/dagger/policy_dagger.pt`
- Command:
  `/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo --config configs/default.yaml --dagger-ckpt logs/dagger/policy_dagger.pt --dagger-data logs/dagger --out logs/ppo_stable_v1`
- Config differences:
  lower PPO learning rate, lower entropy bonus, higher BC auxiliary weight,
  lower initial action std, tanh-squashed Gaussian actions, KL monitoring,
  delayed clip-radius relaxation, richer rollout logging.
- Visual behavior:
  early and mid training produced stable flight; later checkpoints regressed and the final checkpoint crashed into the ground.
- Rollout metrics:
  early updates were healthy at clip radius `0.25`; late updates collapsed at clip radius `1.0` with crash rate `1.0`.
- Evaluation metrics:
  best checkpoint found by direct sweep was `logs/ppo_stable_v1/policy_ppo_0092160.pt`.
  Over 20 evaluation episodes:
  PPO best checkpoint -> completion `100%`, crash `0%`, mean return `70.43`, mean gates `4.00`, first-lap time `499` steps.
  BC baseline -> completion `100%`, crash `0%`, mean return `68.30`, mean gates `4.00`, first-lap time `535` steps.
  Final PPO checkpoint -> completion `0%`, crash `100%`, mean return `-30.92`, mean gates `1.00`.
- Takeaway:
  PPO can improve the BC policy slightly, but the current run over-trains and collapses.
  The right deployment checkpoint is an intermediate PPO checkpoint, not the final one.
- Next action:
  add early-checkpoint selection / model selection by evaluation, then work on generalization across multiple track layouts instead of overfitting one fixed course.

### Run: `ppo_multitrack_v1`

- Date: 2026-04-28
- Goal: Fine-tune PPO on a small distribution of training tracks while validating on held-out tracks and saving the best checkpoint automatically.
- Starting checkpoint: `logs/dagger/policy_dagger.pt`
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo --config configs/multitrack_ppo.yaml --dagger-ckpt logs/dagger/policy_dagger.pt --dagger-data logs/dagger --out logs/ppo_multitrack_v1`
- Training tracks:
  `rect_default`, `rect_wide`, `rect_tall`, `rect_skew`
- Held-out validation tracks:
  `heldout_diamond`, `heldout_tilted`
- Current best checkpoint:
  `logs/ppo_multitrack_v1/policy_ppo_best.pt`
- Best-so-far validation snapshot:
  completion `100%`, mean gates `4.00`, aggregate mean return `70.33`.
- Held-out suite details for current best:
  `heldout_diamond` -> completion `100%`, return `70.23`, finish `497` steps.
  `heldout_tilted` -> completion `100%`, return `70.42`, finish `484` steps.
- BC held-out comparison:
  BC also achieved `100%` completion on the held-out suite, but with lower aggregate mean return `68.80`.
  So the finished multitrack PPO best checkpoint is now outperforming BC on unseen tracks as well.
- Observed behavior so far:
  early validation improved, then later validation began to soften, which confirms that best-checkpoint selection is necessary even in the multitrack setting.

### Multitrack imitation baselines

- `logs/bc_multitrack_v1/policy_bc.pt` on held-out suite:
  aggregate completion `100%`, crash `0%`, mean return `68.80`.
- `logs/dagger_multitrack_v1/policy_dagger.pt` on held-out suite:
  aggregate completion `100%`, crash `0%`, mean return `68.80`.
- Current takeaway:
  multitrack BC and multitrack DAgger both generalize cleanly to the held-out suite, but DAgger has not yet produced a clear held-out advantage over BC by itself.

### Multitrack PPO comparison

- `logs/ppo_multitrack_v1/policy_ppo_best.pt` on held-out suite:
  aggregate completion `100%`, crash `0%`, mean return `70.33`.
  `heldout_diamond` finish `497` steps, `heldout_tilted` finish `484` steps.
- `logs/ppo_multitrack_v2/policy_ppo_best.pt` on held-out suite:
  aggregate completion `100%`, crash `0%`, mean return `70.28`.
  `heldout_diamond` finish `501` steps, `heldout_tilted` finish `495` steps.
- Current champion:
  `logs/ppo_multitrack_v1/policy_ppo_best.pt`
- Current takeaway:
  PPO consistently improves over the multitrack BC/DAgger baselines on unseen tracks, but the newer multitrack-DAgger warm start did not beat the earlier PPO v1 run yet. The difference is small, so we are now in incremental-improvement territory rather than basic stability triage.

### Harder generalization suite baseline

- New config:
  `configs/generalization_hard.yaml`
- Changes:
  larger training track library, three additional harder held-out tracks,
  and randomized start position/yaw relative to the first gate.
- Existing champion before retraining:
  `logs/ppo_multitrack_v1/policy_ppo_best.pt`
- Existing champion on the harder suite:
  aggregate completion `100%`, crash `0%`, mean return `70.51`.
- Multitrack BC baseline on the harder suite:
  aggregate completion `93%`, crash `0%`, mean return `66.81`.
- Current takeaway:
  the current PPO policy is already fairly robust to tougher held-out tracks and randomized starts, while BC degrades more noticeably. The next retraining run is now about chasing incremental generalization gains rather than fixing a collapse.

### Run: `generalization_hard` full pipeline

- Date: 2026-04-29
- Goal: Improve robustness to unseen track geometry and randomized start states without losing stable flight.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_hard.yaml --bc-out logs/bc_generalization_v1 --dagger-out logs/dagger_generalization_v1 --ppo-out logs/ppo_generalization_v1`
- Training distribution:
  `rect_default`, `rect_narrow`, `rect_wide`, `rect_tall`, `rect_skew`, `rect_offset`, `rect_drop`, `rect_bow`, `rect_fast`
- Held-out validation distribution:
  `heldout_diamond`, `heldout_tilted`, `heldout_pinched`, `heldout_zigzag`, `heldout_lowhigh`
- Start randomization:
  reset pose jitter behind the first gate in longitudinal, lateral, vertical, and yaw dimensions.
- BC result:
  `logs/bc_generalization_v1/policy_bc.pt` trained on `48,080` expert transitions.
  Harder-suite aggregate metrics: completion `91%`, crash `0%`, mean gates `3.82`, mean return `66.33`.
- DAgger result:
  `logs/dagger_generalization_v1/policy_dagger.pt` trained on an aggregated dataset of `134,624` transitions.
  Harder-suite aggregate metrics: completion `85%`, crash `0%`, mean gates `3.70`, mean return `64.57`.
- PPO best checkpoint:
  `logs/ppo_generalization_v1/policy_ppo_best.pt`
- PPO harder-suite metrics:
  aggregate completion `100%`, crash `0%`, mean gates `4.18`, mean return `71.68`.
  Per-track returns:
  `heldout_diamond` `72.34`, `heldout_tilted` `70.00`, `heldout_pinched` `76.47`, `heldout_zigzag` `69.44`, `heldout_lowhigh` `70.13`.
- PPO cross-suite check on the earlier multitrack held-out suite:
  aggregate completion `100%`, crash `0%`, mean gates `4.00`, mean return `69.57`.
- Comparison against earlier champion:
  `logs/ppo_multitrack_v1/policy_ppo_best.pt` scored `70.51` on the harder suite and `70.33` on the easier held-out suite.
  `logs/ppo_generalization_v1/policy_ppo_best.pt` improved the harder suite to `71.68` but dropped slightly on the easier suite to `69.57`.
- Takeaway:
  the broader track library plus randomized starts produced the strongest robustness result so far.
  PPO is now clearly ahead of BC and DAgger on hard unseen tracks, but the new champion is slightly less specialized for the easier held-out distribution.
- Next action:
  train with mixed validation or multi-objective checkpoint selection so we do not trade away easy-suite speed while improving hard-suite robustness.

### Run: `ppo_generalization_balanced_v1` (in progress)

- Date: 2026-04-29
- Goal: Recover some easier-suite performance while preserving the harder-suite robustness gained by `ppo_generalization_v1`.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo --config configs/generalization_balanced.yaml --dagger-ckpt logs/dagger_generalization_v1/policy_dagger.pt --dagger-data logs/dagger_generalization_v1 --out logs/ppo_generalization_balanced_v1`
- Training distribution:
  same broader track library and randomized starts as `configs/generalization_hard.yaml`.
- Validation strategy:
  two suites with equal weight in checkpoint selection.
  `easy` uses `heldout_diamond` and `heldout_tilted`.
  `hard` uses the five-track harder suite with randomized starts.
- Early training behavior:
  mixed validation is already surfacing the intended tradeoff rather than hiding it.
  At step `10,240`, the mixed score was completion `90%`, mean return `65.80`, with `easy` return `68.57` and `hard` return `63.03`.
  At step `30,720`, the mixed score improved to completion `95%`, mean return `67.10`, with `easy` return `68.18` and `hard` return `66.02`.
  At step `51,200`, the best-so-far mixed checkpoint reached completion `95%`, mean return `67.64`, with `easy` return `69.06` and `hard` return `66.22`.
- Current takeaway:
  the new validation scheme is working as intended.
  It is selecting checkpoints based on balanced performance instead of letting PPO over-specialize to only the harder suite.

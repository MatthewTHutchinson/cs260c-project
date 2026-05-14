# Progress Log

## Current snapshot

Date: 2026-05-12

Current status:

- The current strongest overall checkpoint is `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`.
- On the richer-observation hard held-out suite, that checkpoint achieved aggregate completion `99%`, crash `0%`, mean gates `4.97`, and mean return `82.49`.
- On the richer-observation easy held-out suite, the same checkpoint achieved aggregate completion `100%`, crash `0%`, mean gates `5.00`, and mean return `82.95`.
- The richer-observation branch is now clearly ahead of every earlier PPO family.
- The first full multimodal run has now completed:
  `logs/ppo_multimodal_obs_v1/policy_ppo_best.pt`
- Best internal validation snapshot for that multimodal run:
  mixed return `72.75`,
  easy return `74.60`,
  hard-core return `74.57`,
  hard-zigzag return `69.35`,
  with mixed completion `100%`.
- Quick 10-episode sanity evaluation of the multimodal PPO best checkpoint under `configs/multimodal_obs_v1.yaml`:
  completion `90%`,
  crash `0%`,
  OOB `0%`,
  mean return `65.69`,
  mean gates `3.80`,
  and mean finish time `530.7` steps.
- Comparison caveat:
  those quick multimodal numbers are not yet a like-for-like replacement for the earlier state-only held-out suite comparisons,
  so they should be treated as promising but not yet definitive.
- The detector-only `vision_bridge` path remains weak:
  the old state-only PPO under `configs/vision_bridge_eval_v1.yaml` still reached `0%` completion and `100%` OOB in the quick bridge eval.
- External source hygiene is now improved:
  `docs/COMPETITION_NOTES.md` is the main source hierarchy for AI Grand Prix / DCL-facing facts,
  anchored on `docs/260508_Technical_Spec_0002.pdf`.
- The older Gmail PDF exports are now stored in `docs/` and should be treated as historical / possibly stale unless the latest spec confirms them.
- The teacher-warmstarted follow-up is now active:
  `configs/multimodal_obs_v2.yaml`
  into
  `logs/bc_multimodal_obs_v2`,
  `logs/dagger_multimodal_obs_v2`,
  and `logs/ppo_multimodal_obs_v2`.
- A competition-spec alignment pass is now also complete:
  the repo has explicit camera intrinsics support,
  spec-aligned bridge and multimodal eval configs,
  and a fail-fast DCL adapter scaffold.

## Immediate next step

Decide whether to:

- launch the teacher-warmstarted follow-up `configs/multimodal_obs_v2.yaml`
- or pause and first tighten the multimodal evaluation story with a cleaner held-out re-evaluation pass

Update:

- the repo now has a completed first end-to-end perception branch:
  onboard RGB rendering in `GateRaceAviary`,
  detector-backed `vision_bridge` observations,
  scene/lighting/domain randomization,
  and multimodal BC/DAgger/PPO support with a state+vision policy.
- the first full run artifacts are now all present in:
  `logs/bc_multimodal_obs_v1`,
  `logs/dagger_multimodal_obs_v1`,
  and `logs/ppo_multimodal_obs_v1`.
- the documentation set is also now organized around:
  `PROGRESS`,
  `REPORT_NOTES`,
  `PORTFOLIO_NOTES`,
  `BRAINSTORMING`,
  and the new `COMPETITION_NOTES`.
- active training has moved on to:
  `train_all.py --config configs/multimodal_obs_v2.yaml`
  and the run has now advanced through BC into DAgger.

Suggested direction:

- treat `generalization_obs_v1` as the new baseline for future work
- use `docs/COMPETITION_NOTES.md` whenever a simulator assumption needs to be compared against the real competition interface
- keep any claim about highlighted gates, no-depth guarantees, or control abstraction tagged as historical unless the latest spec confirms it
- use the observation-upgrade result as a key storyline in the report and portfolio because it is still the clearest architectural win so far

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

### Run: `competition_source_reconciliation`

- Date: 2026-05-12
- Goal:
  reconcile the repo docs against the latest external AI Grand Prix technical spec
  and stop relying on stale brainstorming/email assumptions as if they were current truth.
- Inputs reviewed:
  `docs/260508_Technical_Spec_0002.pdf`
  plus the older Gmail PDF exports now stored in `docs/`.
- Confirmed additions captured:
  MAVLink-over-UDP interface,
  NED/body/camera frame conventions,
  `20`-degree upward camera tilt,
  pinhole intrinsics,
  `640 x 360 @ 30 Hz` JPEG vision stream,
  Windows 11 runtime note,
  and the `8`-minute Round One duration cap.
- Documentation changes:
  added `docs/COMPETITION_NOTES.md`,
  updated `README.md`,
  refreshed the top-level snapshots in the docs,
  and retired the old `brainstorming/` folder after preserving the still-useful points in markdown form.
- Objectivity policy:
  facts are now split into `Confirmed`, `Historical`, and `Inference` buckets.
- Takeaway:
  the project now has a much cleaner separation between
  what the latest spec really says,
  what came from older emails,
  and what is still only a repo-side assumption.

### Run: `multimodal_obs_v1` (completed)

- Date logged: 2026-05-12
- Goal:
  train the first full state+vision branch using the richer `78`-D state vector plus onboard RGB input.
- Final artifacts:
  `logs/bc_multimodal_obs_v1/policy_bc.pt`
  round checkpoints in `logs/dagger_multimodal_obs_v1/`
  and `logs/ppo_multimodal_obs_v1/policy_ppo_best.pt`
- Best internal validation snapshot:
  mixed return `72.75`,
  mixed completion `100%`,
  easy return `74.60`,
  hard-core return `74.57`,
  hard-zigzag return `69.35`,
  mixed mean finish steps `494.87`.
- Quick external sanity evaluation:
  `10` episodes under `configs/multimodal_obs_v1.yaml`
  gave completion `90%`,
  crash `0%`,
  OOB `0%`,
  mean return `65.69`,
  mean gates `3.80`,
  and mean finish time `530.7` steps.
- Important comparison caveat:
  the internal validation and the quick external eval do not yet give a clean one-number replacement for the state-only champion benchmarks.
  This branch looks promising, but not yet clearly like the new overall winner.
- Related negative result that still matters:
  the detector-only `vision_bridge` remained much worse than the multimodal path.
- Takeaway:
  direct multimodal learning appears more viable than trying to reconstruct the old structured observation with a lightweight detector,
  but the evaluation story still needs one cleaner apples-to-apples pass before making strong claims.

### Run: `competition_spec_alignment_update`

- Date: 2026-05-13
- Goal:
  bring the repo's camera/perception assumptions closer to
  `docs/260508_Technical_Spec_0002.pdf`
  without breaking the faster research-oriented training configs.
- Code changes:
  `GateDetector.CameraParams` now defaults to the latest spec's
  `640 x 360` image model with `[fx, fy, cx, cy] = [320, 320, 320, 180]`.
  `GateRaceAviary` now supports explicit camera intrinsics overrides
  plus separate policy-image resizing, so we can render spec-sized frames
  and still feed `128 x 96` multimodal policies.
  `env/dcl_adapter.py` now fails fast instead of silently returning zero telemetry.
- New configs:
  `configs/competition_spec_bridge_eval.yaml`
  and
  `configs/competition_spec_multimodal_eval.yaml`
- Important caveat:
  the published spec's stated intrinsics and stated `VFoV` are not perfectly self-consistent under a standard pinhole model.
  The repo now mirrors the explicit intrinsics directly and keeps that inconsistency flagged in `docs/COMPETITION_NOTES.md`.
- Takeaway:
  the project is still not a finished DCL runtime client,
  but its evaluation path is now closer to the official camera/interface assumptions
  and less dependent on stale historical defaults.

### Run: `robustness_obs_v1`

- Date: 2026-05-03
- Goal: stress-test the current richer-observation PPO champion beyond the standard held-out suites before starting a speed-optimization branch.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.robustness_audit --config configs/robustness_obs_v1.yaml --type ppo --ckpt logs/ppo_generalization_obs_v1/policy_ppo_best.pt --out logs/robustness_obs_v1`
- Audit tooling:
  added `eval/robustness_audit.py`, stress hooks for policy-input observation noise and action noise in `eval.evaluate`, optional disturbance impulses in `GateRaceAviary`, and new OOD audit tracks in `env/tracks.py`.
- Scenario summary:
  `nominal_easy` -> completion `100.0%`, return `83.09`
  `nominal_hard_core` -> completion `99.2%`, return `82.81`
  `nominal_hard_zigzag` -> completion `98.3%`, return `81.39`
  `stress_start_pose` -> completion `98.3%`, return `83.25`
  `stress_obs_noise_light` -> completion `99.2%`, return `82.77`
  `stress_obs_noise_heavy` -> completion `100.0%`, return `83.06`
  `stress_action_noise` -> completion `98.3%`, return `81.44`
  `stress_disturbance` -> completion `0.0%`, return `-50.01`
  `ood_vertical` -> completion `76.7%`, return `60.55`
  `ood_switchback` -> completion `25.0%`, return `33.59`
- Key failure modes:
  `heldout_lowhigh` is the weakest standard held-out track.
  `audit_drop_recover`, `audit_sharp_switchback`, and `audit_offset_spike` are the clearest OOD failures.
  Strong per-step disturbance impulses currently break the controller-policy stack completely.
- Takeaway:
  the richer-observation champion is robust on the known evaluation distribution and reasonably insensitive to the tested observation/action noise levels, but it is not robust yet to harder vertical recovery layouts, sharp switchbacks, or strong external disturbances.
- Next action:
  use the audit results to design the next robustness-improvement branch before any explicit speed-tuning run.

### Run: `vision_perception_branch`

- Date: 2026-05-04
- Goal:
  implement the first realistic perception branch without discarding the strong state-based policies.
- Main code changes:
  `env/gate_race_aviary.py`
  now supports:
  onboard RGB rendering,
  forward camera config knobs,
  detector-backed `vision_bridge` observations,
  scene visuals with gates/floor/walls/clutter,
  and visual domain randomization for lighting, textures, exposure, noise, and occlusion.
  Added texture assets in `assets/textures/`.
- Policy/training changes:
  added multimodal policies in `policy/actor.py`,
  runtime adapters in `policy/runtime.py`,
  and multimodal data flow through
  `training/bc.py`, `training/dagger.py`, `training/ppo.py`,
  `eval/evaluate.py`, `eval/evaluate_track_suite.py`,
  `eval/rank_checkpoints.py`, and `eval/robustness_audit.py`.
- New configs:
  `configs/vision_bridge_eval_v1.yaml`
  for evaluating existing state policies through perception,
  and `configs/multimodal_obs_v1.yaml`
  for state+vision BC -> DAgger -> PPO training.
- Smoke-test status:
  `vision_bridge_eval_v1` env reset/step succeeded with `78`-D observations and live camera detections.
  Multimodal BC forward pass succeeded with image tensor shape `(N, 3, 96, 128)`.
  A tiny multimodal PPO trainer smoke also succeeded after fixing an image-layout bug (`HWC` vs `CHW`) in the dataset path.
- Environment note:
  installed `opencv-python` into the project Conda env because the detector path depends on it and it was missing there.
- Next action:
  launch the first real multimodal training run and compare it against `ppo_generalization_obs_v1`.

### Run: `multimodal_obs_v1` (in progress)

- Date: 2026-05-05
- Goal:
  train the first state+vision policy on top of the richer `78`-D state features,
  using the new onboard camera and visual randomization pipeline.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/multimodal_obs_v1.yaml --bc-out logs/bc_multimodal_obs_v1 --dagger-out logs/dagger_multimodal_obs_v1 --ppo-out logs/ppo_multimodal_obs_v1`
- Config summary:
  state observation remains the richer `78`-D vector,
  plus a `3 x 96 x 128` onboard RGB image branch.
  Training uses randomized textures, lighting, clutter, exposure jitter, image noise, and light occlusions.
- Current status:
  BC finished successfully.
  `logs/bc_multimodal_obs_v1/policy_bc.pt` is written.
  BC dataset sizes:
  `dataset_obs.npy`, `dataset_act.npy`, and `dataset_img.npy` are all present.
  DAgger is active now.
  `logs/dagger_multimodal_obs_v1/policy_dagger_r01.pt` and `policy_dagger_r02.pt` are saved,
  and round 3 is underway.
- Early results:
  the current multimodal BC checkpoint already looks promising.
  On a quick held-out suite sweep:
  aggregate completion `100%`, crash `0%`, mean gates `4.29`, mean return `73.09`.
  Per-track examples:
  `heldout_pinched` return `80.00`,
  `heldout_diamond` up to `77.16`,
  and `heldout_lowhigh` `70.16`.
- Important comparison:
  the old state-only PPO champion performs very poorly when run through the detector-backed perception bridge.
  On `configs/vision_bridge_eval_v1.yaml` over 5 episodes:
  completion `0%`, OOB `100%`, mean return `-14.90`.
  So the perception gap is real, and the multimodal branch is much more promising than relying on detector reconstruction alone.
- Visual fix applied during this run:
  the top and bottom gate rails in the PyBullet scene were rotated correctly so the gate frame orientation now looks right on the previously problematic first and third gates.
- Next action:
  let DAgger finish, then monitor the first PPO validation checkpoints and compare the eventual multimodal best checkpoint against `ppo_generalization_obs_v1`.

### Run: `multimodal_obs_v2` (queued)

- Date: 2026-05-05
- Goal:
  improve the multimodal branch by transferring the strong state-only control prior into the image-conditioned student instead of training the fused policy entirely from scratch.
- Config:
  `configs/multimodal_obs_v2.yaml`
- New training changes:
  multimodal BC now supports:
  state-teacher warm-start,
  and action-level distillation from a state-only teacher during BC and DAgger retraining.
- Teacher choice:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
  as both the warm-start source and distillation teacher.
- Smoke-test result:
  warm-start transfer copied `6` tensors successfully into the multimodal student.
- Queueing plan:
  this run should begin after `multimodal_obs_v1` finishes so the two full pipelines do not overlap on CPU.

### Run: `generalization_robust_obs_v1` (in progress)

- Date: 2026-05-03
- Goal: improve OOD robustness on vertical recovery and sharp switchback layouts without giving back the strong in-distribution gains from `generalization_obs_v1`.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_robust_obs_v1.yaml --bc-out logs/bc_generalization_robust_obs_v1 --dagger-out logs/dagger_generalization_robust_obs_v1 --ppo-out logs/ppo_generalization_robust_obs_v1`
- Training distribution changes:
  added new OOD-inspired but non-held-out layouts:
  `train_vertical_arc_a`, `train_vertical_arc_b`,
  `train_drop_recover_a`, `train_drop_recover_b`,
  `train_switchback_tight_a`, `train_switchback_tight_b`,
  and a softened `train_offset_recover`.
  The training list also upweights several vertical/switchback tracks by repetition.
- Validation changes:
  PPO checkpoint selection now scores five suites:
  `easy`, `hard_core`, `hard_zigzag`, `ood_vertical`, and `ood_switchback`.
  The new OOD suites carry nearly half the aggregate validation weight.
- Reset distribution changes:
  training start jitter is stronger than `generalization_obs_v1`
  to better match the harder audit scenarios.
- Expert sanity check:
  on a quick sweep of representative new training-style tracks,
  the expert completed `train_vertical_arc_a`, `train_drop_recover_a`,
  `train_switchback_tight_a`, and the softened `train_offset_recover`.
- Current status:
  the full BC -> DAgger -> PPO pipeline is running in
  `logs/bc_generalization_robust_obs_v1`,
  `logs/dagger_generalization_robust_obs_v1`,
  and `logs/ppo_generalization_robust_obs_v1`.
- Success criterion:
  beat the current audit baselines on `ood_vertical` and especially `ood_switchback`,
  while keeping standard held-out completion near the current `obs_v1` champion.

### Run: `generalization_robust_obs_v1`

- Date: 2026-05-04
- Final artifacts:
  `logs/bc_generalization_robust_obs_v1/policy_bc.pt`
  `logs/dagger_generalization_robust_obs_v1/policy_dagger.pt`
  `logs/ppo_generalization_robust_obs_v1/policy_ppo_best.pt`
- BC result:
  `60,100` expert transitions, final BC loss about `2.2e-4`.
- DAgger result:
  aggregated dataset grew to `161,068` transitions, final DAgger loss about `1.7e-5`.
- PPO validation:
  `35` validation snapshots were recorded.
  Best saved checkpoint had weighted validation mean return about `72.81`
  with strong OOD switchback validation return about `67.48`.
- Full robustness audit comparison against the previous richer-observation champion:
  `ood_switchback` improved dramatically from completion `25.0%` and return `33.59`
  to completion `90.8%` and return `69.18`.
  Per-track:
  `audit_sharp_switchback` improved from `33.3%` to `95.0%` completion.
  `audit_offset_spike` improved from `16.7%` to `86.7%` completion.
- Vertical OOD result:
  `ood_vertical` improved only slightly overall, from completion `76.7%` / return `60.55`
  to completion `79.2%` / return `61.02`.
  `audit_vertical_ladder` improved strongly, but `audit_drop_recover` remained the main failure case and got slightly worse.
- Standard held-out tradeoff:
  the new robustness branch became more reliable in raw completion on the standard suites,
  but it was slower and lower-return almost everywhere.
  Examples:
  `nominal_easy` return dropped from `83.09` to `82.04`, finish steps worsened from `435.9` to `451.8`.
  `nominal_hard_core` return dropped from `82.81` to `80.20`.
  `nominal_hard_zigzag` return dropped from `81.39` to `74.86`.
- Disturbance stress:
  unchanged failure case.
  Strong per-step disturbance impulses still produced `0%` completion.
- Takeaway:
  the robustness-focused retraining succeeded on the hardest switchback OOD family,
  but it over-specialized and gave up too much nominal-track speed/return while not fully solving the vertical drop/recover case.
- Next action:
  keep both PPO champions for now:
  `ppo_generalization_obs_v1` as the standard-track / speed champion,
  `ppo_generalization_robust_obs_v1` as the switchback-robust champion.
  The next branch should target `audit_drop_recover` specifically while rebalancing validation weights to recover nominal performance.

### Run: `generalization_robust_obs_v2` (in progress)

- Date: 2026-05-04
- Goal: target the remaining `audit_drop_recover` weakness from `robust_obs_v1`
  while protecting more nominal-track performance than the previous robustness branch.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_robust_obs_v2.yaml --bc-out logs/bc_generalization_robust_obs_v2 --dagger-out logs/dagger_generalization_robust_obs_v2 --ppo-out logs/ppo_generalization_robust_obs_v2`
- Training distribution changes:
  added `train_drop_recover_c`, `train_drop_recover_d`, and `train_vertical_ladder_train`.
  The training list now repeats drop/recover and vertical tracks more heavily than switchbacks.
- Validation changes:
  split the former aggregate vertical OOD suite into:
  `ood_vertical_ladder` and `ood_drop_recover`.
  This prevents `audit_vertical_ladder` gains from masking `audit_drop_recover` failures.
  Validation weights are also more balanced:
  `easy 0.20`, `hard_core 0.25`, `hard_zigzag 0.15`,
  `ood_vertical_ladder 0.15`, `ood_drop_recover 0.15`,
  `ood_switchback 0.10`.
- Expert sanity check:
  after softening `train_drop_recover_b` and `train_drop_recover_d`,
  the expert completed representative hard training tracks cleanly, including
  `train_drop_recover_a`, `b`, `c`, `d`,
  `train_vertical_ladder_train`, and `train_switchback_tight_a`.
- Success criterion:
  improve `audit_drop_recover` materially over `robust_obs_v1`
  while keeping more of the nominal return/finish-time profile from `generalization_obs_v1`.

### Run: `generalization_robust_obs_v2`

- Date: 2026-05-04
- Final artifacts:
  `logs/bc_generalization_robust_obs_v2/policy_bc.pt`
  `logs/dagger_generalization_robust_obs_v2/policy_dagger.pt`
  `logs/ppo_generalization_robust_obs_v2/policy_ppo_best.pt`
- Goal:
  rescue the `audit_drop_recover` failure from `robust_obs_v1`
  without giving back as much nominal-track performance.
- Training changes:
  more drop/recover and vertical-ladder style training tracks,
  split validation into `ood_vertical_ladder` and `ood_drop_recover`,
  and reduced `ood_switchback` validation weight.
- Result:
  this branch succeeded on drop/recover but failed overall as a balanced policy.
- Key audit outcomes relative to earlier champions:
  `audit_drop_recover` improved dramatically:
  `obs_v1` `68.3%` completion -> `robust_v1` `58.3%` -> `robust_v2` `100%`.
  `ood_vertical` aggregate improved to `99.2%` completion and return `71.18`.
  But `ood_switchback` collapsed:
  `robust_v1` had `90.8%` completion / return `69.18`,
  while `robust_v2` fell to `24.2%` / `40.47`.
  `audit_sharp_switchback` was especially bad:
  `95.0%` completion in `robust_v1` down to `0%` in `robust_v2`.
- Nominal-track cost:
  this branch also regressed the standard held-out suites further.
  `nominal_easy` return fell to `75.60`
  and finish time slowed to `484.0` steps.
  `nominal_hard_core` return fell to `76.31`.
  `nominal_hard_zigzag` return fell to `69.81`.
- PPO validation interpretation:
  the best saved validation checkpoint itself only had
  `val_ood_switchback_mean_return` about `39.7`,
  so this was not just a bad final-checkpoint selection issue.
- Takeaway:
  `robust_obs_v2` is a negative result in the balanced-policy sense.
  It proves the training/validation emphasis can fully solve `drop_recover`,
  but only by sacrificing both switchback robustness and nominal performance.
- Current champion set:
  keep `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
  as the nominal/speed champion.
  keep `logs/ppo_generalization_robust_obs_v1/policy_ppo_best.pt`
  as the switchback-robust champion.
  Do not promote `robust_obs_v2` as the new default.
- Next action:
  stop pushing the single-policy weighted-mixture approach for now.
  Either:
  train specialized experts/policies per failure family,
  or move to a broader architectural change such as curriculum scheduling,
  conditional policies, or multimodal perception instead of continuing to retune track weights alone.

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

### Run: `ppo_generalization_balanced_v1`

- Date: 2026-05-01
- Final best checkpoint:
  `logs/ppo_generalization_balanced_v1/policy_ppo_best.pt`
- Easier-suite result on `configs/multitrack_ppo.yaml`:
  aggregate completion `100%`, crash `0%`, mean gates `4.50`, mean return `75.15`.
- Harder-suite result on `configs/generalization_hard.yaml`:
  aggregate completion `97%`, crash `0%`, mean gates `4.19`, mean return `71.36`.
- Main weakness:
  `heldout_zigzag` dropped to `85%` completion even though the rest of the suite remained strong.
- Takeaway:
  mixed validation solved the easier-suite tradeoff and produced the best easier-suite checkpoint so far, but it underweighted the most difficult switchback-style track.

### Run: `ppo_generalization_balanced_v2` (in progress)

- Date: 2026-05-01
- Goal: Recover `heldout_zigzag` performance without losing the easier-suite gains from `balanced_v1`.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo --config configs/generalization_balanced_v2.yaml --dagger-ckpt logs/dagger_generalization_v1/policy_dagger.pt --dagger-data logs/dagger_generalization_v1 --out logs/ppo_generalization_balanced_v2`
- Training changes:
  added non-held-out zigzag-like tracks `train_zigzag_a`, `train_zigzag_b`, `train_switchback`, and `train_zigzag_lowhigh`.
  The training track list duplicates the zigzag-like layouts to upweight them during rollout collection.
- Validation changes:
  three weighted suites now drive checkpoint selection:
  `easy` (`0.35`), `hard_core` (`0.30`), and `hard_zigzag` (`0.35`).
- Early validation snapshot:
  step `10,240` reached mixed completion `100%`, mean return `69.46`.
  Per-suite returns:
  `easy` `69.62`, `hard_core` `69.42`, `hard_zigzag` `69.32`.
- Early takeaway:
  the dedicated zigzag suite is no longer trailing immediately, which is a promising sign relative to the previous run.

### Run: `ppo_generalization_balanced_v2`

- Date: 2026-05-01
- Final best checkpoint:
  `logs/ppo_generalization_balanced_v2/policy_ppo_best.pt`
- Easier-suite result on `configs/multitrack_ppo.yaml`:
  aggregate completion `100%`, crash `0%`, mean gates `4.50`, mean return `75.38`.
- Harder-suite result on `configs/generalization_hard.yaml`:
  aggregate completion `100%`, crash `0%`, mean gates `4.11`, mean return `70.93`.
- Key per-track result:
  `heldout_zigzag` recovered to completion `100%` and return `69.64`.
- Comparison:
  this run improved on `ppo_generalization_balanced_v1` by fixing the zigzag failure case and slightly improving the easier suite.
  It still did not beat `ppo_generalization_v1` on the harder-suite aggregate return (`71.68`).
- Takeaway:
  dedicated zigzag weighting was effective for the specific failure mode, but the broader hard-suite optimum still appears to depend on more than just track sampling.

### Run: zigzag-enriched imitation reruns

- Date: 2026-05-01
- Goal: refresh the BC and DAgger warm starts so PPO is not regularized against an older, less zigzag-rich imitation dataset.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_balanced_v2.yaml --bc-out logs/bc_generalization_balanced_v2 --dagger-out logs/dagger_generalization_balanced_v2 --ppo-out logs/ppo_generalization_balanced_v2_from_dagger --skip-ppo`
- BC result:
  `logs/bc_generalization_balanced_v2/policy_bc.pt`
  dataset size `48,080` transitions, final loss `0.00075`.
  Harder-suite aggregate metrics: completion `87%`, crash `0%`, mean gates `3.75`, mean return `65.10`.
- DAgger result:
  `logs/dagger_generalization_balanced_v2/policy_dagger.pt`
  aggregate dataset size `134,624` transitions.
  Harder-suite aggregate metrics: completion `83%`, crash `0%`, mean gates `3.66`, mean return `63.78`.
- Main observation:
  the zigzag-enriched imitation reruns did not beat the earlier generalization imitation baselines, and DAgger remained especially weak on `heldout_lowhigh`.
- Takeaway:
  simply collecting more imitation data on the harder track mix is not enough by itself.
  The expert policy and/or the observation design is likely now the bigger limitation than raw sample count.

### Run: `ppo_generalization_balanced_v3_from_dagger` (in progress)

- Date: 2026-05-01
- Goal: run one final PPO test from the zigzag-enriched DAgger checkpoint to see whether the refreshed imitation prior improves the balanced-v2 PPO result.
- Command:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m training.ppo --config configs/generalization_balanced_v2.yaml --dagger-ckpt logs/dagger_generalization_balanced_v2/policy_dagger.pt --dagger-data logs/dagger_generalization_balanced_v2 --out logs/ppo_generalization_balanced_v3_from_dagger`
- Early validation snapshot:
  step `10,240` reached mixed completion `100%`, mean return `70.03`.
  Per-suite returns:
  `easy` `70.08`, `hard_core` `70.28`, `hard_zigzag` `69.76`.
- Early trend:
  step `20,480` softened to mixed return `69.16`.
  step `30,720` softened further to `68.63`.
  step `40,960` dropped sharply to completion `69%`, with `hard_zigzag` return `49.53`.
- Current takeaway:
  the new DAgger warm start is stable enough to launch PPO, but early evidence does not yet show a clear advantage over `ppo_generalization_balanced_v2`.

### Observation / expert upgrade branch

- Date: 2026-05-02
- Changes implemented:
  `env.gate_race_aviary` now supports configurable observation richness via:
  `lookahead_gates`, `include_gate_normals`, and `include_relative_heading`.
- New richer observation config:
  `configs/generalization_obs_v1.yaml`
- New observation contents:
  body-frame velocity, body-frame angular rate, relative position of the next `3` gates, body-frame normals for the next `3` gates, and a `cos/sin` heading-alignment feature for the next gate plane.
- New observation size:
  single-step observation dim `26`, stacked observation dim `78`.
- Backward compatibility:
  older configs still use the original `36`-D observation and continue to work.
- Expert upgrade:
  `expert.expert_policy` now aims slightly beyond the next gate while blending in the next two gate directions and normals, so it begins turning earlier than a simple gate-center chaser.
- Smoke checks:
  the richer-observation expert completed evaluation cleanly on the named validation suites from `configs/generalization_obs_v1.yaml`.
- Takeaway:
  this is the first meaningful architectural improvement beyond track-distribution tuning.
  The next experiment should retrain BC/DAgger/PPO on `generalization_obs_v1.yaml` rather than continue optimizing the old `36`-D input alone.

### Run: `generalization_obs_v1` full pipeline

- Date: 2026-05-02
- Goal: test whether richer geometric state features plus a stronger lookahead expert materially improve racing performance beyond distribution tuning alone.
- Command:
  `/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_obs_v1.yaml --bc-out logs/bc_generalization_obs_v1 --dagger-out logs/dagger_generalization_obs_v1 --ppo-out logs/ppo_generalization_obs_v1`
- Observation upgrade:
  `3`-gate lookahead, gate normals in body frame, and `cos/sin` heading alignment to the next gate plane.
  Total obs dim: `78`.
- Expert upgrade:
  multi-gate lookahead target point that blends gate normals and future track direction.
- BC result:
  `logs/bc_generalization_obs_v1/policy_bc.pt`
  dataset size `48,080` transitions, final loss `0.00022`.
  Hard-suite aggregate metrics: completion `100%`, crash `0%`, mean gates `4.20`, mean return `72.25`.
- DAgger result:
  `logs/dagger_generalization_obs_v1/policy_dagger.pt`
  aggregate dataset size `134,624` transitions.
  Hard-suite aggregate metrics: completion `100%`, crash `0%`, mean gates `4.26`, mean return `72.95`.
- PPO best checkpoint:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
- PPO easy-suite metrics:
  aggregate completion `100%`, crash `0%`, mean gates `5.00`, mean return `82.95`.
- PPO hard-suite metrics:
  aggregate completion `99%`, crash `0%`, mean gates `4.97`, mean return `82.49`.
- Hard-suite per-track PPO details:
  `heldout_diamond` `83.31`, `heldout_tilted` `83.06`, `heldout_pinched` `83.15`, `heldout_zigzag` `82.20`, `heldout_lowhigh` `80.72`.
- Comparison to previous champion:
  the old stronger hard-suite PPO checkpoint `ppo_generalization_v1` reached return `71.68`.
  `ppo_generalization_obs_v1` improved that to `82.49`.
- Takeaway:
  richer observations and a better expert produced the clearest improvement in the project so far.
  This is a bigger gain than the previous rounds of track-distribution tuning alone.

# CS 260C Project — Drone Gate Racing

## What this is
Hybrid IL + RL for autonomous quadrotor gate racing.
Pipeline: BC → DAgger → PPO fine-tune (single policy, sequential phases).

## Action space
Body-frame waypoint delta [Δx_b, Δy_b, Δz_b, Δψ] — clipped, converted to world-frame target for classical controller.

## Observation space
State-based: body-frame velocity (3), angular rates (3), relative position of next 2 gates in body frame (6). No vision in main loop.

## Simulator
gym-pybullet-drones (installed editable at ../gym-pybullet-drones).
We wrap it with custom gate-event logic: gate_passed, gate_id, wrong_way, collision in info dict.

## Training phases
- Phase B: BC (supervised MSE on expert dataset)
- Phase C: DAgger (aggregate dataset, query expert on policy rollouts)
- Phase D: PPO fine-tune with BC auxiliary loss to prevent forgetting

## Expert
Current code uses a heuristic gate-chasing expert with short lookahead.
Planning notes previously referenced spline/min-snap ideas, but that is not the current implementation.
Expert outputs same action space as learner (body-frame delta).

## Key design choices
- Policy at 10-20 Hz; classical controller runs higher rate
- Body-frame relative gate geometry + 2-gate lookahead
- Action clip: start 0.25m radius, relax with curriculum
- Vision is optional extension (week 5 only)

## Recent changes log
- Added GUI visualization tooling:
  `python3 -m eval.visualize --type expert` now launches PyBullet with real-time playback, optional chase camera, and gate labels/highlighting.
- Expanded local Python requirements to include `matplotlib` and `opencv-python` because the repo already contains plotting and gate-detection utilities that depend on them.
- PPO stabilization pass after BC flew but PPO often crashed into the ground:
  reduced PPO learning rate to `1e-4`, reduced entropy bonus, increased BC auxiliary weight, and added `init_log_std`, `target_kl`, and delayed clip-radius relaxation in `configs/default.yaml`.
- PPO actor now uses a tanh-squashed Gaussian instead of sampling from a Gaussian and clamping afterward. This makes action sampling and log-prob evaluation internally consistent during PPO updates.
- PPO now starts with lower exploration (`init_log_std=-2.0`) so RL fine-tuning stays closer to the BC/DAgger policy early in training.
- PPO rollout logging is richer now: per-update metrics include approximate KL, mean rollout return, mean gates passed, crash rate, and OOB rate. This is meant to make regressions obvious while tuning.
- `eval.plot_metrics` was updated to read both the older array-style PPO metrics and the newer dict-style PPO metrics.
- `eval.rank_checkpoints` was added to sweep PPO checkpoints directly in the simulator and find the best deployable checkpoint.
- `eval.evaluate` now measures first-lap completion time correctly instead of using episode end as a proxy for finish time.
- Added named track layouts in `env/tracks.py` plus reset-time track sampling in `GateRaceAviary`, so training configs can now use multi-track distributions.
- Added `configs/multitrack_ppo.yaml` with train-track sampling and held-out validation tracks.
- Added `configs/generalization_hard.yaml` with a larger track library, harder held-out layouts, and randomized starts.
- PPO now supports periodic validation and saves `policy_ppo_best.pt` based on validation performance instead of only relying on the final checkpoint.
- Added `eval.evaluate_track_suite` for per-track held-out evaluation.
- `GateRaceAviary.reset()` now supports randomized start pose jitter relative to the first gate via config keys for longitudinal, lateral, vertical, and yaw jitter.
- The harder-distribution training run `logs/ppo_generalization_v1` produced the current best robustness checkpoint:
  `logs/ppo_generalization_v1/policy_ppo_best.pt`.
  It reached aggregate return `71.68` with `100%` completion on the harder held-out suite in `configs/generalization_hard.yaml`.
- That same checkpoint still scores aggregate return `69.57` with `100%` completion on the earlier easier held-out suite in `configs/multitrack_ppo.yaml`, while `logs/ppo_multitrack_v1/policy_ppo_best.pt` remains slightly better there at `70.33`.
- Current interpretation:
  broader track diversity plus randomized starts improved robustness on hard unseen tracks, but there is now a measurable specialization tradeoff between the easier and harder validation suites.
- PPO now supports `validation_suites` in config, which allows multiple named validation environments with weighted aggregate checkpoint selection.
- `configs/generalization_balanced.yaml` uses that feature to score checkpoints equally on an easier held-out suite and the harder randomized-start suite.
- `logs/ppo_generalization_balanced_v1/policy_ppo_best.pt` became the best easier-suite specialist so far:
  aggregate return `75.15` on `configs/multitrack_ppo.yaml`, but only `97%` completion on the harder suite because `heldout_zigzag` dropped to `85%` completion.
- Added four zigzag-style training layouts:
  `train_zigzag_a`, `train_zigzag_b`, `train_switchback`, and `train_zigzag_lowhigh`.
- Added `configs/generalization_balanced_v2.yaml`, which:
  upweights zigzag-like training tracks via repeated sampling
  and uses three validation suites: `easy`, `hard_core`, and `hard_zigzag`.
- `logs/ppo_generalization_balanced_v2/policy_ppo_best.pt` is now the strongest easier-suite checkpoint so far:
  aggregate return `75.38` on `configs/multitrack_ppo.yaml`.
  It also restored `heldout_zigzag` to `100%` completion on the harder suite, but its harder-suite aggregate return is still `70.93`, below `ppo_generalization_v1` at `71.68`.
- Zigzag-enriched BC and DAgger reruns were completed into:
  `logs/bc_generalization_balanced_v2` and `logs/dagger_generalization_balanced_v2`.
  Those imitation checkpoints did not outperform the earlier generalization imitation baselines on the harder held-out suite.
- A final PPO comparison run is in progress at:
  `logs/ppo_generalization_balanced_v3_from_dagger`
  using `configs/generalization_balanced_v2.yaml` but warm-starting from `logs/dagger_generalization_balanced_v2/policy_dagger.pt`.
- Early validation on that final PPO test has been stable but not obviously stronger than `ppo_generalization_balanced_v2`.
  The first checkpoint reached mixed return `70.03`, but later early checkpoints softened and by step `40,960` mixed completion had fallen to `69%`.
- Current interpretation:
  track weighting helped PPO fix the zigzag-specific failure mode, but the expert policy and/or observation design is now a more likely bottleneck than raw imitation sample count.
- A richer-observation branch is now implemented in `env/gate_race_aviary.py`:
  `lookahead_gates`, `include_gate_normals`, and `include_relative_heading` are configurable per env config.
- `configs/generalization_obs_v1.yaml` is the new richer-state training config.
  It uses:
  lookahead `3`, gate normals enabled, and explicit heading-alignment features.
  Resulting obs dim is `78` (`26` per frame x `3` stacked frames).
- `expert/expert_policy.py` now uses multi-gate lookahead and gate-normal-aware pass-through targeting instead of just chasing the next gate center.
- Training/eval entrypoints now validate that `policy.obs_dim` matches the env observation size, which should prevent silent shape mismatches on future observation changes.
- `eval.evaluate_track_suite` now understands `validation_suites` configs, not just `validation_env`.
- The full `generalization_obs_v1` pipeline was trained successfully.
  Current best checkpoint:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
  Hard-suite aggregate: completion `99%`, return `82.49`
  Easy-suite aggregate: completion `100%`, return `82.95`
- The richer-observation BC and DAgger baselines also improved substantially:
  `logs/bc_generalization_obs_v1/policy_bc.pt` -> hard return `72.25`
  `logs/dagger_generalization_obs_v1/policy_dagger.pt` -> hard return `72.95`
- This is currently the clearest project conclusion:
  richer observations and a better expert delivered a larger gain than additional track-resampling and curriculum tuning by themselves.
- Added first-pass robustness audit support:
  `eval/robustness_audit.py`, `configs/robustness_obs_v1.yaml`,
  policy-input observation noise and action noise support in `eval.evaluate`,
  optional disturbance impulses in `GateRaceAviary`,
  and new OOD audit tracks in `env/tracks.py`.
- Audit results for `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`:
  `nominal_easy` completion `100.0%`
  `nominal_hard_core` completion `99.2%`
  `nominal_hard_zigzag` completion `98.3%`
  `stress_start_pose` completion `98.3%`
  `stress_action_noise` completion `98.3%`
  `ood_vertical` completion `76.7%`
  `ood_switchback` completion `25.0%`
  `stress_disturbance` completion `0.0%`
- Current interpretation:
  the richer-observation champion is strong on the known held-out distribution and mild noise sweeps, but the next real robustness gap is OOD geometry, especially sharp switchbacks and vertical drop/recover layouts.
- Added a new robustness-focused training config:
  `configs/generalization_robust_obs_v1.yaml`
  It keeps the richer `78`-D observation branch but expands the training distribution with:
  `train_vertical_arc_a`, `train_vertical_arc_b`,
  `train_drop_recover_a`, `train_drop_recover_b`,
  `train_switchback_tight_a`, `train_switchback_tight_b`,
  and `train_offset_recover`.
- PPO validation in that config now scores five suites:
  `easy`, `hard_core`, `hard_zigzag`, `ood_vertical`, and `ood_switchback`.
- A quick expert sanity check confirmed completion on representative new training-style tracks after softening `train_offset_recover`.
- The full pipeline is now running via:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_robust_obs_v1.yaml --bc-out logs/bc_generalization_robust_obs_v1 --dagger-out logs/dagger_generalization_robust_obs_v1 --ppo-out logs/ppo_generalization_robust_obs_v1`
- That run finished.
  BC collected `60,100` transitions.
  DAgger reached `161,068` transitions.
  PPO best checkpoint:
  `logs/ppo_generalization_robust_obs_v1/policy_ppo_best.pt`
- Robustness audit result for that checkpoint:
  `ood_switchback` improved massively from completion `25.0%` to `90.8%`.
  `ood_vertical` improved only slightly from `76.7%` to `79.2%`.
  `stress_disturbance` remained `0%` completion.
- Standard held-out tradeoff:
  this robustness-focused branch is slower and lower-return than `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
  on the standard held-out suites, even though completion became cleaner.
- Current interpretation:
  there are now effectively two useful PPO champions:
  `ppo_generalization_obs_v1` for stronger nominal/easier performance,
  and `ppo_generalization_robust_obs_v1` for switchback OOD robustness.
  The next robustness branch should focus on `audit_drop_recover` while rebalancing validation so nominal speed does not regress as much.
- That next branch is now `configs/generalization_robust_obs_v2.yaml`.
  Main changes:
  more drop/recover and vertical-ladder style training tracks,
  separate validation suites for `ood_vertical_ladder` and `ood_drop_recover`,
  and reduced `ood_switchback` validation weight.
- Quick expert sanity checks were run before training,
  and the expert completed the representative hard training-style tracks after softening `train_drop_recover_b` and `train_drop_recover_d`.
- Active run:
  `KMP_DUPLICATE_LIB_OK=TRUE /Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python train_all.py --config configs/generalization_robust_obs_v2.yaml --bc-out logs/bc_generalization_robust_obs_v2 --dagger-out logs/dagger_generalization_robust_obs_v2 --ppo-out logs/ppo_generalization_robust_obs_v2`
- `generalization_robust_obs_v2` is now complete and should be treated as a negative result for balanced performance.
  It solved `audit_drop_recover`:
  completion reached `100%` in the audit.
  But it lost the switchback gains from `robust_obs_v1`:
  `ood_switchback` fell back to `24.2%` completion.
  It also further regressed nominal held-out return and finish time.
- PPO metrics confirm this was not just a bad final checkpoint:
  the best validation snapshot still had `val_ood_switchback_mean_return` only about `39.7`.
- Current interpretation:
  reweighting the single-policy training distribution has likely hit diminishing returns.
  The next meaningful branch should be something structurally different:
  specialized policies, curriculum scheduling, conditional control, or perception/architecture changes.
- That perception branch is now implemented:
  `GateRaceAviary` supports onboard RGB rendering, visible gates/floor/walls/clutter, and visual randomization.
  `observation_source="vision_bridge"` reconstructs gate-relative observations from the onboard camera via `GateDetector`.
  This is intentionally a hybrid bridge: dynamics are still from state, while gate positions come from perception.
- Added multimodal state+vision policies in `policy/actor.py`
  plus shared routing helpers in `policy/runtime.py`.
  BC, DAgger, PPO, and the eval scripts now support either state-only or multimodal policies from config.
- New configs:
  `configs/vision_bridge_eval_v1.yaml`
  for testing existing MLP checkpoints through perception,
  and `configs/multimodal_obs_v1.yaml`
  for the first full state+vision training branch.
- Smoke tests completed:
  detector-backed env reset/step worked,
  multimodal BC forward/training worked,
  and a tiny multimodal PPO rollout/update smoke worked.
  One bug was fixed during this pass:
  image datasets now normalize to `CHW` consistently across BC, DAgger, and PPO.
- Current live multimodal run:
  `train_all.py --config configs/multimodal_obs_v1.yaml`
  BC is complete, DAgger is in progress, and `policy_dagger_r01.pt` plus `policy_dagger_r02.pt` are already written.
- Quick evaluation signals:
  `logs/bc_multimodal_obs_v1/policy_bc.pt` achieved aggregate held-out return `73.09` with `100%` completion in a short suite sweep.
  By contrast, the detector-backed `vision_bridge` baseline for the old state-only PPO is poor:
  `0%` completion and `100%` OOB over 5 episodes.
- A follow-up config is now prepared:
  `configs/multimodal_obs_v2.yaml`
  It adds:
  `warmstart_state_ckpt`,
  `warmstart_state_type`,
  `distill_teacher_ckpt`,
  `distill_teacher_type`,
  and `distill_coef`.
  The intended teacher is `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`.
- On this machine, some training/eval runs may need:
  `KMP_DUPLICATE_LIB_OK=TRUE`
  to work around a duplicate OpenMP runtime issue in the Conda env.
- Added `docs/COMPETITION_NOTES.md` and moved the latest external reference PDF to:
  `docs/reference/260508_Technical_Spec_0002.pdf`
  This is now the main competition-facing source hierarchy and confidence tracker.
- The old `brainstorming/` folder has been retired after promoting the still-useful content into the `docs/` markdown files.
- `multimodal_obs_v1` has completed end to end.
  Artifacts now exist in:
  `logs/bc_multimodal_obs_v1`,
  `logs/dagger_multimodal_obs_v1`,
  and `logs/ppo_multimodal_obs_v1`.
- `multimodal_obs_v2` has now been started directly via:
  `train_all.py --config configs/multimodal_obs_v2.yaml`
  writing into
  `logs/bc_multimodal_obs_v2`,
  `logs/dagger_multimodal_obs_v2`,
  and `logs/ppo_multimodal_obs_v2`.
  Latest observed state:
  BC completed and DAgger is currently active.
- Competition-spec alignment pass completed:
  `GateDetector.CameraParams` now defaults to the latest spec's `640 x 360` / `[320, 320, 320, 180]` camera model,
  `GateRaceAviary` now supports explicit camera intrinsics plus separate policy-image resizing,
  and new evaluation configs exist at
  `configs/competition_spec_bridge_eval.yaml`
  and
  `configs/competition_spec_multimodal_eval.yaml`.
- `env/dcl_adapter.py` now fails fast instead of silently returning zero telemetry / fake observations.
  It is still a scaffold, not a working competition client.
- Training is now more restart-friendly:
  `train_all.py --resume` skips completed BC/DAgger/PPO stages when artifacts already exist,
  DAgger persists `resume_dataset_*.npy` plus per-round checkpoints,
  and PPO writes `trainer_state_latest.pt` for update-level restart.
- On macOS, use:
  `caffeinate -dimsu -w <train_all_pid>`
  during long runs.
  This prevents idle sleep, but not shutdown, reboot, or closing the lid.
- There is now a stronger generalization audit beyond the old held-out suite:
  `configs/extended_generalization_obs_eval.yaml`
  tests mirrored/right-turn tracks plus longer `6`-gate courses.
- That audit exposed a real blind spot in the current state champion:
  it is still strong on the legacy held-out family,
  but it weakens on mirrored/right-turn tracks and fails most longer-course tracks.
- `expert/expert_policy.py` has been upgraded to use a longer-horizon future-gate blend.
  This improved mirrored/right-turn and `long_hex` behavior,
  but `long_snake` remains a stretch audit family rather than a good current imitation target.
- Prepared next state-training branch:
  `configs/generalization_bidirectional_obs_v1.yaml`
  for richer-state training on mirrored/right-turn plus solvable longer-course layouts.
- `scripts/continue_training.sh` now automates the next long-run sequence:
  wait for `multimodal_obs_v2`,
  resume it if needed,
  evaluate the finished multimodal PPO checkpoint,
  launch `generalization_bidirectional_obs_v1`,
  then evaluate the finished bidirectional state branch.
- Mid-run sanity signal:
  `logs/dagger_multimodal_obs_v2/policy_dagger_r03.pt`
  reached `100%` completion and mean return `77.34`
  on the original multimodal held-out suite,
  so the active multimodal run is still worth finishing.
- There is now a matching harder multimodal audit config:
  `configs/extended_generalization_multimodal_eval.yaml`
  so the multimodal branch can be tested on mirrored/right-turn and longer-course layouts,
  not just the original multimodal held-out family.
- `scripts/training_status.sh` prints the current active processes,
  latest checkpoint artifacts,
  and continuation-log tail for long-run monitoring.

## Notes for future Codex sessions
- For external AI Grand Prix / DCL facts, start with:
  `docs/COMPETITION_NOTES.md`
  and only then consult
  `docs/reference/260508_Technical_Spec_0002.pdf`.
- Treat the Gmail PDF exports in `docs/` as historical context, not primary truth.
- Do not rely on the old `brainstorming/` directory; it has been retired.
- The published spec's `[fx, fy]` and `VFoV` are not perfectly self-consistent.
  Prefer explicit intrinsics when mirroring the competition camera in code.
- If BC works but PPO regresses immediately, inspect PPO rollout crash rate and mean gates before assuming the reward is wrong.
- The current expert implementation is still a simple gate-chasing heuristic in code, even though earlier planning notes mention spline/min-snap.
- Use the Conda interpreter at `/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python` if the shell `python3` is missing project dependencies.
- Documentation workflow now lives in `README.md` plus:
  `docs/PROGRESS.md`, `docs/REPORT_NOTES.md`, `docs/PORTFOLIO_NOTES.md`, `docs/BRAINSTORMING.md`, and `docs/COMPETITION_NOTES.md`.
- After the zigzag-enriched imitation reruns, decide whether to spend another PPO run on the new DAgger checkpoint or move next to richer observations / stronger expert logic.
- The next major training branch after the current PPO comparison should be `generalization_obs_v1`.
- `generalization_obs_v1` is now the main champion branch to build on.
- Before the next speed-optimization push, use `logs/robustness_obs_v1/report.md` and `per_track.csv` to target new robustness work at `heldout_lowhigh`, `audit_drop_recover`, `audit_sharp_switchback`, and `audit_offset_spike`.
- The immediate active follow-up run after the first robustness audit is `generalization_robust_obs_v1`.
- That follow-up run is complete; the next most likely branch is a drop/recover-focused robustness v2.

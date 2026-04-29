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
- A new run is in progress at `logs/ppo_generalization_balanced_v1`.
  Early mixed-validation checkpoints are healthy, with the best-so-far at step `51,200` reaching mixed completion `95%` and mixed mean return `67.64`.

## Notes for future Claude sessions
- If BC works but PPO regresses immediately, inspect PPO rollout crash rate and mean gates before assuming the reward is wrong.
- The current expert implementation is still a simple gate-chasing heuristic in code, even though earlier planning notes mention spline/min-snap.
- Use the Conda interpreter at `/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python` if the shell `python3` is missing project dependencies.
- Documentation workflow now lives in `README.md` plus:
  `docs/PROGRESS.md`, `docs/REPORT_NOTES.md`, and `docs/PORTFOLIO_NOTES.md`.
- After `ppo_generalization_balanced_v1` finishes, compare its best checkpoint on both `configs/multitrack_ppo.yaml` and `configs/generalization_hard.yaml` before promoting it.

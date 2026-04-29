# Report Notes

## Working thesis

A staged imitation-to-RL pipeline is a practical way to bootstrap autonomous drone racing because pure RL exploration is too unstable early on, while BC and DAgger provide a strong flight prior that PPO can refine.

## Draft report structure

1. Problem statement
2. Environment and task setup
3. Observation and action design
4. Expert policy and imitation learning
5. DAgger aggregation
6. PPO fine-tuning and stabilization
7. Evaluation results
8. Limitations and future work

## Key implementation details to explain

- Observation:
  body-frame velocity, angular rates, and next-two-gate relative positions
- Action:
  body-frame waypoint delta plus yaw delta
- Environment:
  custom gate events on top of `gym-pybullet-drones`
- Reward:
  sparse gate reward, crash penalty, velocity-alignment shaping, jerk penalty
- Curriculum:
  clip radius increases during PPO

## Evidence to collect

- BC visual success examples
- DAgger dataset size growth
- PPO before-vs-after stabilization behavior
- evaluation metrics for expert, BC, and PPO
- training curves from `eval.plot_metrics`

## Claims to verify before writing final prose

- Did PPO improve lap speed, gates passed, or return over BC?
- Did PPO retain controllability after the stabilization changes?
- Did BC already solve most of the task, leaving PPO mainly to optimize speed?
- Which failure modes remain: ground crash, gate miss, oscillation, or OOB?

## Verified findings so far

- BC is already a strong baseline on the default rectangular track: `100%` completion and `0%` crash rate in evaluation.
- Stabilized PPO produced useful intermediate checkpoints.
- Best PPO checkpoint identified so far:
  `logs/ppo_stable_v1/policy_ppo_0092160.pt`
- That checkpoint outperformed BC on both mean return and first-lap completion time while preserving perfect completion on the default track.
- The final PPO checkpoint was worse than earlier checkpoints, so model selection matters; "last checkpoint" is not a safe proxy for "best policy" here.
- In the multitrack setting, the best saved PPO checkpoint generalizes cleanly to held-out tracks and currently outperforms BC on unseen-track aggregate return.
- Comparing two multitrack PPO runs, `ppo_multitrack_v1` is still the best overall held-out performer, though `ppo_multitrack_v2` remains very close. This suggests the main gains now depend more on fine optimization than on fixing a broken pipeline.
- A harder generalization configuration with more training tracks and randomized starts produced a new robustness champion:
  `logs/ppo_generalization_v1/policy_ppo_best.pt`.
- On the harder held-out suite, that checkpoint achieved `100%` completion, `0%` crash rate, and aggregate mean return `71.68`, outperforming both `bc_generalization_v1` (`66.33`) and `dagger_generalization_v1` (`64.57`).
- The new generalization champion still transfers back to the earlier held-out suite with `100%` completion and aggregate mean return `69.57`, though the older easier-suite specialist `ppo_multitrack_v1` remains slightly better there at `70.33`.
- This gives a useful report narrative:
  broader training distributions and randomized starts improved robustness to harder unseen tracks, while slightly reducing specialization on the easier validation distribution.
- Mixed validation is now implemented in PPO, so the next report-quality comparison can ask whether multi-objective checkpoint selection reduces that tradeoff.

## Limitations to mention honestly

- current expert is a heuristic gate chaser, not a full minimum-snap planner
- DCL adapter is still a stub and not yet competition-ready
- training currently uses state observations, not end-to-end vision
- PPO remains sensitive to exploration and reward design
- validation choice now matters strategically, because the "best" checkpoint depends on whether the metric emphasizes easy-track speed or hard-track robustness

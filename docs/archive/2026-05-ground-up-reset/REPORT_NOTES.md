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
  body-frame velocity, angular rates, and relative gate geometry in the body frame
- Action:
  body-frame waypoint delta plus yaw delta
- Control abstraction caveat:
  the waypoint delta is a repo-side learning abstraction, not a confirmed competition command.
  In PyBullet it becomes a nearby position target for the PID controller.
  A competition-facing adapter should likely map it into `SET_POSITION_TARGET_LOCAL_NED`
  first as position setpoints, then as position setpoints with velocity feedforward.
- Environment:
  custom gate events on top of `gym-pybullet-drones`
- Reward:
  sparse gate reward, crash penalty, velocity-alignment shaping, jerk penalty
- Curriculum:
  clip radius increases during PPO
- Perception branch:
  onboard RGB rendering, detector-backed gate estimation, and a first multimodal state+vision policy path

## Competition-facing constraints to state clearly

- Latest external source of truth:
  `docs/reference/260508_Technical_Spec_0002.pdf`
  (`VADR-TS-002`, Issue `00.02`, dated `2026-05-08`)
- Official interface summary from that spec:
  MAVLink 2 over UDP,
  deterministic simulator,
  `120 Hz` physics,
  `640 x 360 @ 30 Hz` forward camera stream,
  and an `8`-minute maximum Round One duration.
- Coordinate/frame details worth explaining carefully:
  NED convention,
  body frame `X` forward / `Y` right / `Z` down,
  and camera sharing the body origin with a `20`-degree upward tilt.
- Official runtime note:
  Windows 11 is the expected simulator platform and Linux is currently unsupported in the spec.
- Camera-model nuance to mention honestly:
  the published intrinsics and published `VFoV` are not perfectly self-consistent,
  so the repo now mirrors the explicit intrinsics directly and keeps that discrepancy flagged.

## Claims to mark as preliminary or historical

- Older Gmail PDFs suggested highlighted or guided VQ1 gates; that is useful context but not the latest ground truth.
- Older Gmail PDFs said the API would not expose depth, engine RPM, or battery state of charge; keep those claims marked as historical unless a newer spec restates them.
- Older emails used throttle / roll / pitch / yaw wording for control,
  but the latest spec explicitly lists `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`,
  so the old control wording should be treated as preliminary rather than authoritative.
- The repo does not yet implement the final DCL MAVLink / UDP client path end to end, so do not overstate competition readiness.
- The DCL adapter file is now explicitly fail-fast rather than silently behaving like a working client.
- The current waypoint-delta action is not the final MAVLink interface.
  The latest spec confirms `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`,
  so deployment requires an adapter that decides whether the policy maps to position,
  velocity, acceleration, or attitude targets.

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
- A richer-observation branch is now implemented:
  the next policy can observe three gates ahead, gate-normal orientation features, and explicit heading alignment to the next gate plane.
- The expert has also been upgraded from a simple gate-center chase to a multi-gate lookahead targeter.
- This creates a clean report narrative transition from distribution tuning to representation design.
- That richer-observation branch produced the strongest result in the project so far:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`.
- On the richer-observation hard held-out suite, PPO reached aggregate completion `99%` and mean return `82.49`.
- On the richer-observation easy held-out suite, PPO reached aggregate completion `100%` and mean return `82.95`.
- The upgraded imitation baselines improved too:
  richer-observation BC reached hard-suite return `72.25`, and richer-observation DAgger reached `72.95`.
- This is a strong argument that representation design and a better expert mattered more than additional track-resampling alone.
- A dedicated robustness audit was added and run on the richer-observation champion:
  `eval/robustness_audit.py` with outputs in `logs/robustness_obs_v1/`.
- On that audit, the current best PPO remained strong on the known distribution:
  `100.0%` completion on `nominal_easy`,
  `99.2%` on `nominal_hard_core`,
  and `98.3%` on `nominal_hard_zigzag`.
- The audit also exposed the next honest limitation:
  performance drops sharply on harder OOD layouts, especially `ood_vertical` (`76.7%` completion) and `ood_switchback` (`25.0%` completion).
- Mild observation noise and moderate action noise did not materially degrade performance in this first audit, so the bigger remaining weakness appears to be geometric OOD generalization rather than small Gaussian sensor noise.
- A follow-up robustness-focused training branch is now running:
  `configs/generalization_robust_obs_v1.yaml`
  -> `logs/ppo_generalization_robust_obs_v1`.
- That branch expanded the training distribution with vertical-recovery and switchback-style tracks that resemble the audit failures without reusing the exact held-out OOD layouts.
- It also changed PPO checkpoint selection to score five suites, including explicit `ood_vertical` and `ood_switchback` validation sets.
- The result was a clear win on the switchback OOD family:
  audit `ood_switchback` improved from completion `25.0%` / return `33.59`
  to completion `90.8%` / return `69.18`.
- But the same branch slightly worsened most standard held-out returns and finish times,
  and only marginally improved the vertical OOD aggregate because `audit_drop_recover` remained difficult.
- This is a strong report-quality example of a distribution-shift tradeoff:
  targeted robustness training can solve one OOD family while still over-specializing and sacrificing nominal speed.
- A follow-up `generalization_robust_obs_v2` branch is now running to test a more surgical intervention:
  separate validation for `audit_drop_recover`,
  more drop/recover-inspired training tracks,
  and reduced switchback validation weight so nominal-track performance is better protected.
- That `robust_obs_v2` branch produced an important negative result:
  it fully solved `audit_drop_recover` in the audit,
  but `ood_switchback` collapsed back down to roughly the weak `obs_v1` regime.
- This is good report material because it shows that even fairly sophisticated distribution reweighting hit a limit:
  one-policy track-distribution tuning alone was not enough to simultaneously optimize nominal speed,
  switchback OOD robustness, and drop/recover OOD robustness.
- A new perception branch is now implemented in code:
  the simulator can render onboard RGB frames, randomize the scene visually,
  and reconstruct approximate gate-relative observations through `GateDetector`.
- A first multimodal training run is also now underway:
  `configs/multimodal_obs_v1.yaml`
  -> `logs/ppo_multimodal_obs_v1`.
  BC already completed and DAgger is in progress.
- Early multimodal result:
  the first multimodal BC checkpoint achieved `100%` completion and aggregate return `73.09`
  on a quick held-out suite sweep, which is a strong sign that adding images did not break the existing richer-state policy structure.
- Very important perception result:
  the old state-only PPO champion performs poorly when its structured gate inputs are replaced by the detector-backed `vision_bridge`.
  That bridge baseline reached `0%` completion and `100%` OOB over 5 episodes on `vision_bridge_eval_v1`.
  This is strong evidence that end-to-end or multimodal perception is necessary; simply reconstructing the old observation format with a lightweight detector is not enough.
- A follow-up `multimodal_obs_v2` config is now prepared with state-teacher warm-start and distillation from the current best state-only PPO checkpoint.
  This is a principled next step for the report because it reframes multimodal learning as transfer learning rather than cold-start replacement.
- This gives the report a strong “future work became implementation” arc:
  the project no longer only discusses vision as a next idea;
  it now has an actual bridge and multimodal training path, even though the best completed results are still from the richer state-based PPO branch.
- The first full multimodal PPO run is now completed:
  `logs/ppo_multimodal_obs_v1/policy_ppo_best.pt`
  with best internal mixed validation return `72.75`,
  easy `74.60`,
  hard-core `74.57`,
  and hard-zigzag `69.35`.
- Important objectivity note:
  the multimodal branch looks promising, but a quick external `10`-episode sanity eval under `configs/multimodal_obs_v1.yaml`
  only reached `90%` completion and `65.69` mean return.
  That means the multimodal result is encouraging, but it is not yet a clean new champion claim.
- The new `docs/COMPETITION_NOTES.md` file is useful report scaffolding in its own right,
  because it cleanly separates latest-spec facts from older email assumptions and repo-side inferences.
- A new extended audit now makes the robustness story more honest:
  once mirrored/right-turn and longer `6`-gate tracks were added,
  the current stronger state-based champion no longer looked uniformly robust.
  It stayed strong on the old held-out family,
  but degraded on mirrored/right-turn layouts and failed most of the harder longer-course family.
- This is important report material because it shows the earlier benchmark was directionally informative but incomplete.
  The project did not "solve racing" in a broad sense;
  it solved a narrower but still meaningful family of simulator courses.
- The mirrored/right-turn gap appears to be a real policy-generalization issue.
  By contrast, some of the longer-course failures also exposed a limitation in the heuristic expert,
  which is why the expert was upgraded to use a longer future-gate blend before scheduling the next bidirectional training branch.
- Operationally, long-running laptop training needed explicit resilience work:
  the repo now supports stage-aware `train_all.py --resume`,
  round-level DAgger resume,
  and PPO trainer-state resume.
  This is practical infrastructure rather than algorithmic novelty, but it matters for reproducibility.
- A first speed/control branch now exists in code:
  `GateRaceAviary` supports `control_mode: position_velocity`, which keeps the same 4-D policy action
  but passes capped velocity feedforward into the PID controller.
  Directly evaluating the existing state champion under this mode caused crashes,
  so this should be treated as a retraining branch rather than a plug-in speed boost.

## Limitations to mention honestly

- current expert is a heuristic gate chaser, not a full minimum-snap planner
- DCL adapter is still a stub and not yet competition-ready
- current local waypoint control is stable but conservative,
  and likely caps racing speed compared with velocity, acceleration, or attitude-target control
- best completed results currently still use structured state observations rather than end-to-end vision
- PPO remains sensitive to exploration and reward design
- validation choice now matters strategically, because the "best" checkpoint depends on whether the metric emphasizes easy-track speed or hard-track robustness
- even the new best PPO checkpoint is still state-based and simulator-privileged, so its performance should not be conflated with real-world camera-only racing capability
- the new robustness audit shows that strong results on held-out in-distribution tracks do not yet imply robustness to sharper OOD switchbacks, vertical drop/recover layouts, or external disturbances
- even after targeted OOD retraining, vertical drop/recover layouts remain a meaningful failure mode

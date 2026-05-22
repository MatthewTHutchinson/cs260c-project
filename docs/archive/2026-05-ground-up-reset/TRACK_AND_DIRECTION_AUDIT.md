# Track and Project Direction Audit

Date: 2026-05-22

## Bottom line

The current project has learned useful gate-following behavior, but the track suite is not yet trustworthy enough to drive the next major research claim or long training run.

The biggest issue is not one bad track. It is that track design, validation, and project direction have been evolving together. Several "held-out" and "OOD" tracks are close variants of training tracks, while the next planned bidirectional branch is not runnable as written because it mixes 4-gate and 6-gate layouts in one sampled training environment.

## Evidence

Static audit command:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_tracks.py --out logs/track_audit
```

Manual expert sweep used 3 episodes per track, small start jitter, richer-state observation settings, and the current `ExpertPolicy`.

## Track Library Findings

- `TRACK_LIBRARY` currently has 46 layouts.
- Gate counts are heavily skewed: 39 are 4-gate loops and only 7 are 6-gate layouts.
- Direction is also skewed: 37 are counter-clockwise and 9 are clockwise.
- Most layouts are hand-authored perturbations of the same small rectangular/diamond coordinate box.
- All layouts are cyclic loops inferred from gate centers, but the competition notes describe a start gate, sequential race gates, and a finish gate.

This means the current policy can look robust while still learning a family-specific loop-following behavior.

## Validation Leakage

Nearest held-out/audit tracks are often extremely close to training-style tracks under cyclic/reversed center alignment:

| Held-out or audit track | Nearest train-like track | RMSD |
| --- | --- | ---: |
| `audit_vertical_ladder` | `train_vertical_ladder_train` | 0.166 m |
| `audit_drop_recover` | `train_drop_recover_c` | 0.166 m |
| `heldout_long_hex` | `train_long_hex_b` | 0.258 m |
| `heldout_long_snake` | `train_long_snake` | 0.258 m |
| `heldout_zigzag` | `train_zigzag_lowhigh` | 0.278 m |
| `heldout_right_lowhigh` | `train_right_lowhigh` | 0.278 m |
| `heldout_lowhigh` | `train_right_lowhigh` | 0.296 m |

This does not make those evaluations useless, but it does mean they should be called interpolation/near-family tests, not OOD robustness.

## Config Findings

`configs/generalization_bidirectional_obs_v1.yaml` is currently invalid for training. Its main `env.track_names` mixes 4-gate and 6-gate tracks, but `GateRaceAviary._validate_track_options()` requires all sampled tracks in one environment to have the same number of gates.

Several training configs intentionally repeat track names to weight the sampler. That is fine as an experiment, but it should be represented as explicit weights in a manifest rather than hidden as duplicate names. Hidden duplicates make later audits harder to trust.

## Expert Solvability Findings

The current expert completed most 4-gate layouts, but the sweep found weak or failed tracks:

| Track | Gates | Expert completion |
| --- | ---: | ---: |
| `train_long_snake` | 6 | 0% |
| `heldout_long_snake` | 6 | 0% |
| `heldout_long_snake_right` | 6 | 0% |
| `train_switchback_tight_b` | 4 | 33% |
| `heldout_long_hex_right` | 6 | 33% |
| `audit_sharp_switchback` | 4 | 33% |

Because BC and DAgger depend on expert labels, unsolved training tracks can poison imitation data. These tracks should either be audit-only until the expert is improved, or the expert/controller needs to be upgraded before using them for training.

## Simulator/Competition Alignment Gaps

- Gate crossing uses a circular radius check, while the spec describes a 1.5 m x 1.5 m inner square opening.
- Gate frame collision is not modeled; the visual frame is not a physical obstacle.
- Current tracks do not include confirmed course features such as explicit start/finish gates, obstacles, boundary elements, and environmental structures.
- State policies receive exact relative gate centers and normals. That is useful for research, but it is privileged relative to the competition interface.
- `vision_bridge` can fall back to track geometry (`cache_track`), which leaks ground-truth gate positions when detection fails.
- Most training runs use 20 Hz policy control and PyBullet at 240 Hz; the competition notes confirm 120 Hz physics, 30 Hz camera, and MAVLink command rate below 100 Hz.
- The current action is a body-frame waypoint delta tracked by PID, not yet a real MAVLink `SET_POSITION_TARGET_LOCAL_NED` or `SET_ATTITUDE_TARGET` runtime policy.

## Direction Audit

The project should stop treating incremental PPO track reweighting as the main path forward. The recent branches show the pattern clearly:

- Richer observations plus a better expert gave the largest real gain.
- Track reweighting fixed one audited failure at a time, but often traded off another.
- Robustness v1 improved switchbacks; robustness v2 improved drop/recover but lost switchbacks and nominal speed.
- The next bidirectional config was prepared, but it cannot run as written.

The right next move is benchmark hygiene first, then structural improvements.

## Recommended Next Work

1. Freeze a clean benchmark hierarchy.
   - `dev_train`: tracks allowed for training.
   - `dev_val`: visible validation for checkpoint selection.
   - `frozen_test`: not used for tuning.
   - `stress_ood`: named stress families, never silently moved into training.

2. Replace hand-authored near-duplicates with a seeded procedural track generator.
   - Store metadata for gate count, direction, turn severity, vertical severity, path length, and seed.
   - Reject tracks that are too similar to train tracks.
   - Reject or quarantine tracks the expert cannot solve.

3. Convert track definitions from cyclic loops to explicit race courses.
   - Start gate.
   - Ordered intermediate gates.
   - Finish gate.
   - Completion terminates or freezes score at finish.

4. Fix the gate model.
   - Use square gate-open checks or at least compare circular vs square scoring.
   - Add physical collision for gate frames and optional boundaries/obstacles.
   - Keep official gate dimensions as defaults.

5. Repair the bidirectional plan before launching it.
   - Either split 4-gate and 6-gate training into separate phases/environments, or update the environment/training loop to sample variable gate counts safely.
   - Do not train on `train_long_snake` until expert completion improves.

6. Make perception benchmarks honest.
   - Add a no-ground-truth perception mode where missing detections fall back to cache or zeros, not track positions.
   - Evaluate state-only, vision-bridge, and multimodal policies on the same frozen track suite.
   - Add a blind/no-image ablation for multimodal policies to confirm the image branch is doing work.

7. Move control toward the competition interface.
   - Keep the waypoint-delta policy as a stable baseline.
   - Add a competition-aligned action head for velocity/acceleration/yaw-rate or attitude targets.
   - Evaluate at command rates compatible with the spec.

## Current Run State

- No training process is active.
- `multimodal_obs_v2` has completed BC and DAgger artifacts.
- `logs/ppo_multimodal_obs_v2` has no PPO artifacts yet.
- `logs/continue_training.log` indicates the previous continuation attempt failed around `stdbuf` on macOS.

Recommendation: do not launch another long training run until the track suite and bidirectional config are cleaned up.

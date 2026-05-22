# VQ1 Environment and Strategy Discussion

Date: 2026-05-22

## Purpose

This is the working strategy document for the post-midterm reset.

The short version: VQ1 is close, but it is not a panic sprint. If it opens late next week and remains open for a month or more, we have enough time to make structural changes. The main risk is spending that time polishing the wrong simulator assumptions.

This document should stay editable. It is meant to hold the argument about what to fix first, what to keep as baselines, and what work should wait until we have real VQ1 feedback.

## Current Thesis

The project should pivot from "make the current policy better on our hand-authored tracks" to "make the simulator and evaluation harness honest enough that policy improvements mean something."

The current state champion is useful. The multimodal branch is useful. The expert and track library are useful scaffolding. But the environment is now the bottleneck:

- tracks are too hand-shaped and too near-family
- the task is cyclic loops, while VQ1 is a start/intermediate/finish racecourse
- gate scoring and collisions are simplified
- some state observations are privileged relative to the competition interface
- the next planned bidirectional branch is invalid as written
- the expert fails on several tracks we were starting to treat as training targets

So the next serious work should begin at the environment and benchmark layer.

## Timing Assumption

Working assumption:

- Today is Friday, 2026-05-22.
- VQ1 opens late next week, approximately 2026-05-28 to 2026-05-29.
- VQ1 remains open for at least a month.

This timing is a project assumption, not a confirmed line from `docs/COMPETITION_NOTES.md`. If a newer official date appears, update this section first.

## What We Know from the Spec

Source: `docs/COMPETITION_NOTES.md`.

Confirmed constraints that should shape the environment:

- physics update frequency is `120 Hz`
- camera stream is `30 Hz`
- camera resolution is `640 x 360`
- camera/body share the same origin
- camera is tilted `20` degrees upward
- pinhole camera intrinsics are stated as `[fx, fy, cx, cy] = [320, 320, 320, 180]`
- no GPS and no exposed absolute global position
- telemetry includes attitude, orientation, linear velocities, and system status flags
- MAVLink messages include `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`
	- https://mavlink.io/en/guide/mavlink_2.html
- the course includes a start gate, intermediate gates, a finish gate, obstacles, boundaries, terrain, and environmental structures
- maximum run duration is `8` minutes

The current PyBullet task matches some of this conceptually, but not enough mechanically.

## Strategy Reset

We should run two workstreams in parallel once VQ1 opens:

1. Real VQ1 client/runtime bring-up.
   The first priority after access opens is not training. It is observing the real interface, receiving images and telemetry, sending safe commands, logging everything, and discovering what the course actually looks like.

2. Simulator environment repair.
   While the client work proceeds, the PyBullet environment should be made less self-deceptive: explicit racecourses, square gates, collisions, honest perception modes, and audited train/validation/test splits.

The simulator should become a tool for rapid iteration after we learn from VQ1, not a substitute reality.

## Workstream A: Environment Repair

### A1. Replace cyclic tracks with explicit racecourses

Current model:

```text
gate_0 -> gate_1 -> ... -> gate_n -> gate_0
```

Needed model:

```text
start_gate -> race_gate_1 -> ... -> race_gate_n -> finish_gate
```

Design notes:

- Completion should be based on crossing the finish gate after the ordered gate sequence.
- Evaluation should report first-finish time and stop counting repeated laps unless explicitly configured.
- Training can still use loop mode for old baselines, but new VQ1-aligned configs should default to finite courses.
- Course metadata should include gate count, direction family, vertical severity, turn severity, path length, seed, and split assignment.

### A2. Fix gate geometry and collision

Current issue:

- gate crossing uses a circular radius check
- official inner opening is square: `1.5 m x 1.5 m`
- visual gate frames do not block the drone

Needed changes:

- Add square-opening gate crossing.
- Add optional gate frame collision bodies.
- Keep the old circular scoring behind a config flag for backward comparison.
- Add boundary and obstacle primitives, even if simple at first.

The goal is not perfect DCL replication. The goal is to remove obviously friendly scoring.

### A3. Build procedural track/course generation

The hand-authored library should stop being the main source of truth.

Generator requirements:

- seeded output
- explicit split assignment: `dev_train`, `dev_val`, `frozen_test`, `stress_ood`
- minimum distance from frozen test to train courses
- left/right balance
- 4-gate, 6-gate, and longer-course families
- vertical families: ladder, drop/recover, low-high
- turn families: sweep, zigzag, switchback
- rejection if gates overlap, path segments are degenerate, or expert sanity check fails

The audit script in `scripts/audit_tracks.py` should become part of this workflow.

### A4. Make perception modes honest

Current issue:

`vision_bridge` can fall back to track geometry. That makes debugging easier, but it can hide perception failure.

Needed modes:

- `state`: privileged state baseline
- `vision_bridge_track_fallback`: debugging only
- `vision_bridge_cache_only`: uses detections and temporal cache, no new track truth
- `vision_bridge_zeros`: harsh failure mode
- `multimodal`: state/telemetry plus image, with ablations

Every perception result should report detection count, fallback count, and confidence distribution.

### A5. Match spec timing defaults

New VQ1-aligned environment defaults should move toward:

- PyBullet/controller physics cadence near `120 Hz`
- policy/control command rate below `100 Hz`
- camera render cadence `30 Hz`
- competition camera defaults: `640 x 360`, explicit intrinsics, `20` degree upward tilt
- optional policy-image resizing for neural policies

We can still train faster-than-real-time, but the simulated timing should not quietly teach an impossible control loop.

## Workstream B: Runtime Bring-Up

As soon as VQ1 opens, the first deliverable should be a logging client, not a racing policy.

Minimum client milestones:

1. Connect to MAVLink over UDP.
2. Maintain heartbeat.
3. Receive and timestamp telemetry.
4. Receive, reassemble, decode, and save JPEG camera stream.
5. Send a benign command safely.
6. Record synchronized logs: telemetry, images, command outputs, simulator time.
7. Run the gate detector offline on captured frames.
8. Build a playback harness so captured VQ1 data can be used without the simulator running.

The repo already has `env/dcl_adapter.py`, but it is correctly marked as a scaffold. This work should make that scaffold real.

## Workstream C: Policy Strategy

### Keep current champions as baselines

Keep these as reference points:

- `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
- `logs/ppo_generalization_robust_obs_v1/policy_ppo_best.pt`
- `logs/ppo_generalization_robust_obs_v2/policy_ppo_best.pt`
- `logs/dagger_multimodal_obs_v2/policy_dagger.pt`

Do not discard them. They tell us what the old environment could teach.

### Stop launching big PPO runs until benchmarks are cleaned up

PPO should wait until:

- track/course splits are explicit
- invalid mixed-gate-count configs are fixed
- expert sanity checks pass for training courses
- frozen validation/test sets are not near-duplicates of training courses
- perception fallback mode is explicit in the config name

### Upgrade the expert before using harder courses for imitation

The current expert is still a geometric lookahead policy. It is good enough for many short loops, but it fails some long/switchback tracks.

Next expert direction:

```text
course gates -> racing line -> time allocation -> smooth trajectory -> local target/action labels
```

The expert does not need to be perfect, but it must be reliable on any course used for BC/DAgger. Failed expert tracks should be audit-only.

### Decide what the policy should ultimately output

Current action:

```text
[dx_body, dy_body, dz_body, dyaw]
```

This is stable and worth keeping as a baseline. But VQ1-aligned work should test at least one command-compatible alternative:

- body-frame velocity plus yaw-rate
- local-NED position plus velocity feedforward
- attitude target plus thrust or vertical velocity

The practical target should be whichever maps cleanly to `SET_POSITION_TARGET_LOCAL_NED` or `SET_ATTITUDE_TARGET` after runtime testing.

## Immediate Priority List

### Before VQ1 Opens

1. Fix the benchmark plan.
   - Create explicit course split manifests.
   - Quarantine failed-expert tracks from training.
   - Fix or replace `configs/generalization_bidirectional_obs_v1.yaml`.

2. Implement racecourse mode.
   - Start/intermediate/finish gates.
   - Finite completion.
   - Backward-compatible loop mode.

3. Implement square gate scoring.
   - Compare old circular scoring versus square scoring.
   - Update evaluation reports to include scoring mode.

4. Add no-ground-truth perception evaluation.
   - Make fallback mode visible in config and logs.
   - Add detection/fallback metrics.

5. Prepare VQ1 client scaffolding.
   - MAVLink heartbeat/telemetry loop.
   - UDP camera receiver skeleton.
   - Log format and playback harness.

### First 48 Hours After VQ1 Opens

1. Run the real simulator with no learning assumptions.
2. Capture representative camera and telemetry logs.
3. Verify frame conventions and command behavior.
4. Test whether gates are visually detectable with the current detector.
5. Measure latency, image frequency, telemetry frequency, and command response.
6. Update this document with observed facts.
7. Decide whether the near-term VQ1 policy should be:
   - heuristic perception plus controller
   - learned state/telemetry policy
   - multimodal policy
   - hybrid planner/controller

### After Initial VQ1 Bring-Up

1. Rebuild simulator courses using observed VQ1 geometry/style.
2. Train only after the new benchmark harness is in place.
3. Compare:
   - old state champion
   - robust state champion
   - multimodal DAgger/PPO
   - heuristic baseline
   - upgraded expert/controller
4. Use frozen test only for milestone decisions.

## Decision Points

Open questions for us:

- Should finite racecourse mode replace loop mode everywhere, or only in new configs?
- Should the next environment patch support variable gate counts in one sampled env, or should we split training phases by gate count?
- How much should we invest in PyBullet visuals before seeing actual VQ1 frames?
- Should the first VQ1 submission prioritize reliable completion over speed, even if the path is conservative?
- Should we train a new policy from scratch after environment repair, or distill from current champions into the repaired environment?
- What is the simplest command interface that behaves predictably in the real DCL simulator?

## Working Recommendation

My current recommendation:

1. Make a competition-aligned environment branch first.
2. Keep current champions frozen as baselines.
3. Bring up the VQ1 runtime as soon as access opens.
4. Do not do another large PPO run on the current hand-authored track suite.
5. Use the first real VQ1 observations to decide whether the next big branch is perception, control, or trajectory/expert.

The project is not behind because the midterm exposed flaws. This is exactly the useful failure mode: the old simulator was good enough to prove the pipeline can fly gates, and now we know which assumptions have to be retired before the result matters.

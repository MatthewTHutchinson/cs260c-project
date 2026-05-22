# Ground-Up Restart Plan

Date: 2026-05-22

## Reset Decision

We are not optimizing around the old policies anymore.

Past policies and training logs are now baselines and historical evidence, not design constraints. The project restarts around the competition problem:

```text
recognize gates
control the drone
plan and navigate the course
```

## Archived Docs

Stale midterm-era documents were moved to:

```text
docs/archive/2026-05-ground-up-reset/
```

The raw reference PDFs remain in:

```text
docs/reference/
```

## New Source-of-Truth Docs

- `docs/COMPETITION_CONTEXT.md`
- `docs/GATE_RECOGNITION.md`
- `docs/DRONE_CONTROL.md`
- `docs/PATH_PLANNING_NAVIGATION.md`
- `docs/GROUND_UP_RESTART.md`

These are now the active project docs.

## New Architecture

Target architecture:

```text
VQ1 simulator
  -> FPV frame receiver
  -> telemetry receiver
  -> gate recognition
  -> temporal tracking
  -> navigation state machine
  -> local guidance command
  -> drone control adapter
  -> MAVLink command sender
```

The old PyBullet learner can be rebuilt underneath this architecture, but it should not define the architecture.

## Development Priorities

### Priority 1: VQ1 Runtime Bring-Up

When access opens:

1. connect
2. receive telemetry
3. receive camera frames
4. send safe commands
5. log everything
6. replay logs offline

No policy training is needed for this milestone.

### Priority 2: Gate Recognition

Build a simple detector using real VQ1 frames.

Start classical:

- color/contrast segmentation
- contour/quadrilateral fitting
- confidence scoring
- bearing and range estimate
- temporal tracking

Only add learned perception if the real frames defeat the simple detector.

### Priority 3: Reactive Completion Baseline

Build a non-learning baseline:

```text
gate image error -> local velocity/yaw command -> control adapter
```

This gives us a working completion candidate and a debugging tool for every later learned policy.

### Priority 4: Simulator Rebuild

Fix PyBullet to match the new task:

- finite start/intermediate/finish courses
- square gate scoring
- gate frame collision
- boundaries and simple obstacles
- VQ1 camera timing and intrinsics
- no ground-truth perception fallback for competition-facing configs

### Priority 5: Learning, Only After the Stack Works

Learning can return after:

- real VQ1 logs exist
- the reactive baseline works in simulation
- control adapter behavior is known
- simulator courses are finite and audited
- gate detection has measurable reliability

Candidate learning uses:

- imitation from the reactive/planning stack
- policy refinement around the navigation state
- learned perception only if classical detection is insufficient

## What Not To Do Next

- Do not launch another big PPO run on the old hand-authored tracks.
- Do not use cyclic loop performance as a competition claim.
- Do not rely on exact simulator gate positions.
- Do not assume throttle/roll/pitch/yaw is the final wire-level interface despite the April 10 wording.
- Do not build a complex neural detector before inspecting real VQ1 images.

## Next Concrete Tasks

1. Implement VQ1 logging client skeleton.
2. Implement camera packet receiver and frame writer.
3. Implement MAVLink heartbeat/telemetry receiver.
4. Implement command probe scripts.
5. Prioritize MAVSDK Offboard attitude-rate/thrust probing.
6. Keep body-velocity/yaw-rate as the fallback probe.
7. Refactor PyBullet into finite-course mode.
8. Add square gate scoring.
9. Add detector/tracker message types.
10. Build the reactive completion baseline.

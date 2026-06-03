# Ground-Up Restart Plan

Date: 2026-05-22

This is the active project planning and direction document. If a single doc needs to answer "what are we building next and why?", use this one.

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
- `docs/VQ1_ACCESS_RUNBOOK.md`
- `docs/GROUND_UP_RESTART.md`

These are now the active project docs.

## Current Technical Recommendation

The current recommendation is not a video world model and not raw-pixel RL.

The VQ1 stack should start with:

```text
classical gate recognition
  -> temporal tracking
  -> reactive navigation
  -> MAVSDK Offboard attitude-rate/thrust commands
```

Then add learning only after the baseline works:

```text
gate bearing/size/confidence/history
attitude/orientation
linear velocity if exposed
HIGHRES_IMU acceleration and angular rates if reliable
  -> compact policy
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

Rationale:

- VQ1 is completion-first, not fastest-time-first.
- The spec gives FPV plus telemetry, including `ATTITUDE`, `HIGHRES_IMU`, and linear velocity.
- MAVSDK is the correct first client layer.
- `AttitudeRate + thrust` maps cleanly to `SET_ATTITUDE_TARGET` and the "Pilot Commands -> Stabilized Controller" architecture.
- A compact policy over detector/tracker features is feasible on modest compute.
- Heavy world models, raw video RL, and full NMPC are too expensive for the near-term solo-team path.

Body-velocity plus yaw-rate remains a fallback control probe if attitude-rate/thrust is unstable or poorly supported by the simulator.

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

The old PyBullet learner is quarantined under `legacy/pybullet/`. It should not define the architecture.

## Development Priorities

### Priority 1: VQ1 Runtime Bring-Up

When access opens:

1. connect
2. receive telemetry
3. confirm `ATTITUDE`, `HIGHRES_IMU`, velocity, and status fields
4. receive camera frames
5. send safe commands
6. log everything
7. replay logs offline

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

### Priority 4: Local Simulation Harness

Use the local simulation environment as the primary harness while Windows/VQ1 access is unavailable:

- use the stable no-FPV Mac smoke path for Betaflight/control checks
- keep the stock harness as a known baseline
- use the `AutonomousRacingPilot` simulator adapter
- use the interactive editor path for FPV detector tuning
- quarantine world pose from the algorithm
- export run telemetry/video for presentation evidence

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
- Do not use old PyBullet track performance as a competition claim.
- Do not rely on exact simulator gate positions.
- Do not assume throttle/roll/pitch/yaw is the final wire-level interface despite the April 10 wording.
- Do not build a complex neural detector before inspecting real VQ1 images.

## Next Concrete Tasks

1. Implement VQ1 logging client skeleton.
2. Implement camera packet receiver and frame writer.
3. Implement MAVLink heartbeat/telemetry receiver.
4. Implement command probe scripts.
5. Secure a Windows execution path using `docs/VQ1_ACCESS_RUNBOOK.md`.
6. Prioritize MAVSDK Offboard attitude-rate/thrust probing.
7. Keep body-velocity/yaw-rate as the fallback probe.
8. Use `scripts/run_elodin_smoke.sh` as the repeatable no-FPV control check.
9. Run `RACE_SOLVER=solver.cs260c_pilot scripts/run_elodin_smoke.sh` after controller changes.
10. Tune the detector on local-simulation FPV frames through the interactive simulator.
11. Build the reactive completion baseline around search, track, commit, and reacquire.

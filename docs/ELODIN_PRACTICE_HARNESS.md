# Elodin Practice Harness Direction

Date: 2026-05-31

## Decision

Use the Elodin AI Grand Prix practice harness as the primary local simulator while Windows access to the official VQ1 simulator is unavailable.

PyBullet remains useful for quick unit tests and old-policy comparisons, but it should no longer be treated as the main race-world authority. The previous PyBullet work had weak spots in exactly the places that matter for the final project: track layout quality, gate geometry, rendering realism, and UI/debug ergonomics.

## Why Elodin First

Elodin is a better local-first fit because it gives us:

- macOS/Linux support
- a polished editor UI with FPV and chase views
- a race-oriented harness instead of a generic gym task
- `640 x 360` FPV rendering with spec-style intrinsics
- camera tilt aligned to the current spec
- real Betaflight SITL in the loop
- a single `autopilot(update) -> RCCommand` solver boundary
- database export for telemetry, camera streams, and run review

That solves a lot of the quality problem that came from hand-authoring our own world in PyBullet.

## What Elodin Does Not Solve

Elodin is still not the official qualifier simulator.

Known caveats:

- It uses Betaflight UDP/RC commands rather than MAVSDK/MAVLink commands.
- It exposes ENU world state, while the official spec uses NED conventions.
- It exposes world pose to the solver, while the official spec says absolute global position is not exposed.
- It is a three-gate practice course, not necessarily the VQ1 course.
- Atmospheric effects and race environment complexity are simplified.

These are manageable only if we keep a strict algorithm boundary.

## Non-Negotiable Boundary

The algorithm must not consume world pose.

Allowed inputs:

- FPV camera frame
- body-frame IMU
- attitude/orientation
- angular rates
- linear velocity, if exposed by the target runtime
- barometer/status fields, if useful

Forbidden competition-facing inputs:

- global/world position
- simulator gate IDs
- hard-coded gate positions
- depth
- map truth

Elodin world pose is allowed for:

- scoring
- plotting
- debugging
- sanity checks

It is not allowed in `AutonomousRacingPilot.update(...)`.

## Integration Plan

Target repo layout:

```text
/Users/matthewhutchinson/dev/cs260c-project
  algorithm/
  docs/

/Users/matthewhutchinson/dev/elodin-ai-grand-prix
  solver/
  sim/
  betaflight/
```

Keep the Elodin repo as a sibling checkout. Do not vendor it into this course repo.

Integration path:

1. Bring up the stock Elodin harness on macOS.
2. Confirm the baseline solver runs and exports a run database.
3. Inspect `solver/api.py` and `solver/baseline.py`.
4. Add an Elodin solver that imports or mirrors the `AutonomousRacingPilot` interface.
5. Convert Elodin `SensorUpdate` into our allowed telemetry/frame inputs.
6. Convert `RacingCommand` into Elodin `RCCommand`.
7. Log gate estimates, controller modes, and commands.
8. Tune the detector/controller against Elodin FPV before touching more PyBullet tracks.

## Final Project Framing

The final project should say:

> We develop a simulator-independent autonomous drone racing algorithm and validate it in a higher-quality local harness with real Betaflight SITL and FPV rendering.

It should not say:

> We fixed our project management process.

The presentation can mention PyBullet as an early prototype that exposed simulator-quality risk, then shift to Elodin as the local harness used to evaluate the final algorithm.

## Immediate Setup Checklist

Local prerequisites:

- Xcode Command Line Tools: installed
- `uv`: installed through Homebrew
- `git-lfs`: installed through Homebrew and initialized
- Elodin CLI/editor: installed in `~/.cargo/bin`
- Elodin DB CLI: installed in `~/.cargo/bin`
- cloned `elodin-sys/ai-grand-prix`: `/Users/matthewhutchinson/dev/elodin-ai-grand-prix`
- Betaflight submodule fetched and built

Healthy run signal:

```text
SUCCESS: SITL integration working! Drone took off!
[RACE] ...
```

First deliverable after bring-up:

```text
Elodin stock baseline run + exported FPV video + CSV telemetry
```

Second deliverable:

```text
Elodin solver using our gate detector/tracker/controller boundary without world pose
```

Initial solver adapter:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix/solver/cs260c_pilot.py
```

Run it with:

```bash
RACE_SOLVER=solver.cs260c_pilot elodin editor sim/main.py
```

## Local Bring-Up Status

Completed on 2026-05-31:

- `uv sync` completed in the Elodin repo.
- `uv run pytest` passed: `36 passed`.
- Betaflight SITL built successfully at:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix/betaflight/obj/main/betaflight_SITL.elf
```

- A fast no-FPV smoke run completed with successful takeoff:

```text
SUCCESS: SITL integration working! Drone took off!
[RACE] course=easy gates_passed=0/3 lap_time=2.00s status=DNF pass_times=[--,--,--]
```

- The smoke database was exported to CSV:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix/dbs/betaflight_db001-csv/
```

The no-FPV control smoke path is now reliable through this repo's wrapper:

```bash
scripts/run_elodin_smoke.sh
```

That wrapper uses an inline Betaflight subprocess from the Elodin harness instead of the normal `elodin run` s10 handoff. It is the right command for checking that physics, Betaflight lockstep, RC commands, and our solver adapter are wired together.

The full FPV/render/editor path is separate. Do not enable FPV in `scripts/run_elodin_smoke.sh`; direct inline mode can hang at the first render request because there is no editor/render-server lifecycle. Use the Elodin editor command for visual gate-recognition inspection:

```bash
scripts/run_elodin_editor.sh
```

By default, that wrapper writes an algorithm-only trace here:

```text
logs/elodin_pilot_trace_editor.csv
```

It also saves fresh FPV frames here:

```text
logs/elodin_fpv_frames/
```

Run detector inspection on those frames with:

```bash
scripts/inspect_gate_frames.py \
  --source logs/elodin_fpv_frames \
  --out-dir logs/gate_inspection_elodin \
  --save-mask

scripts/plot_pilot_trace.py \
  --trace logs/elodin_pilot_trace_editor.csv \
  --out logs/elodin_editor_trace.png
```

Validation rule:

- If `elodin_pilot_trace_editor.csv` is mostly `search` while the gate is visibly in `logs/elodin_fpv_frames`, the CV detector is not tuned to the Elodin gate material.
- If the trace enters `detected`/`commit` but the drone flies away or oscillates, the next problem is control sign/gain tuning or FOV-aware navigation.
- If the baseline solver sees gates poorly, do not treat that as our autonomy result. The baseline uses privileged gate positions and can pitch forward without preserving visual gate centering.

- `solver/cs260c_pilot.py` now imports the active `algorithm/` package and maps `AutonomousRacingPilot` output to Elodin `RCCommand` without passing world pose into the pilot.

Experimental smoke command from this repo:

```bash
scripts/run_elodin_smoke.sh
```

Verified no-FPV smoke commands:

```bash
scripts/run_elodin_smoke.sh
RACE_SOLVER=solver.cs260c_pilot scripts/run_elodin_smoke.sh
```

Current caveat: the no-FPV control path is solved enough for local bring-up, but FPV detection tuning still needs the interactive editor/render path or a separate render-server workflow.

Frame inspection command:

```bash
scripts/inspect_gate_frames.py --demo --out-dir logs/gate_inspection_demo --save-mask
```

For real Elodin/VQ1 frames, point `--source` at a frame directory or video and inspect the saved overlays plus `trace.csv`.

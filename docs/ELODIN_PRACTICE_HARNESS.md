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

It captures the Elodin/editor terminal output here:

```text
logs/elodin_editor_stdout.log
```

It also saves fresh FPV frames here:

```text
logs/elodin_fpv_frames/
```

By default, the wrapper clears the previous editor trace/frame/log outputs before
launching so stale files do not get mixed with the current run. Set
`CLEAR_EDITOR_LOGS=0` if you intentionally want to append a manual debugging
session without clearing old artifacts.

The wrapper also clears stale Elodin editor/render-server/Betaflight processes
before launch and logs both the Elodin CLI version and Python package version.
This matters because the FPV camera path depends on the `0.17.3` sensor-camera
API, where frames are published to DB messages by the render server.

Run detector inspection on those frames with:

```bash
scripts/audit_elodin_editor_run.py

scripts/inspect_gate_frames.py \
  --source logs/elodin_fpv_frames \
  --out-dir logs/gate_inspection_elodin \
  --save-mask

scripts/plot_pilot_trace.py \
  --trace logs/elodin_pilot_trace_editor.csv \
  --out logs/elodin_editor_trace.png
```

Run the same saved-frame inspection with a GateNet-style ONNX export:

```bash
CS260C_GATE_DETECTOR=gatenet \
CS260C_GATE_DETECTOR_MODEL=models/gatenet.onnx \
CS260C_GATE_DETECTOR_OUTPUT=corners8 \
scripts/inspect_gate_frames.py \
  --source logs/elodin_fpv_frames \
  --out-dir logs/gatenet_inspection_elodin
```

Use this offline step before live flight. The current course-completion results
still use the classical HSV detector because no GateNet weights/export are
checked into this repo.

Run a selected simulator course with:

```bash
ELODIN_COURSE=easy ELODIN_SIM_TIME=12 scripts/run_elodin_editor.sh
```

Available local validation courses in the sibling harness patch:

- `easy`: three straight gates; current champion completes this course
- `lateral_soft`: small lateral offsets after gate 0
- `low_high`: modest gate-height variation after gate 0
- `four_gate_straight`: longer straight-line completion check
- `circular_arc`: four yawed gates on a gentle left-turning arc
- `s_curve`: five yawed gates with linked left/right lateral wiggles

Run a suite with:

```bash
scripts/run_elodin_course_suite.py --courses easy,lateral_soft,low_high,four_gate_straight
```

Compare camera assumptions across the suite:

```bash
scripts/run_elodin_course_suite.py \
  --courses easy,lateral_soft,low_high,four_gate_straight \
  --camera-profiles vq1_pinhole,gatenet_fisheye
```

Run the hard presentation stress suite with:

```bash
scripts/run_elodin_course_suite.py \
  --courses circular_arc,s_curve \
  --out-dir logs/elodin_hard_track_results \
  --sim-time 16 \
  --timeout-s 360 \
  --idle-timeout-s 70 \
  --frame-stride 8
```

The suite writes per-course traces/logs/frames under:

```text
logs/elodin_course_suite/
```

Validation rule:

- If `elodin_pilot_trace_editor.csv` is all `search`, `frame_fresh` is always
  `0`, and `logs/elodin_fpv_frames/` is empty, the solver is not receiving FPV
  frames from the Elodin editor/render path. Check
  `logs/elodin_editor_stdout.log` for `[FPV] First frame...`, render-trigger
  failures, or collect errors before tuning CV.
- If `elodin_pilot_trace_editor.csv` is mostly `search` while the gate is visibly in `logs/elodin_fpv_frames`, the CV detector is not tuned to the Elodin gate material.
- If the trace enters `detected`/`commit` but the drone flies away or oscillates, the next problem is control sign/gain tuning or FOV-aware navigation.
- If the baseline solver sees gates poorly, do not treat that as our autonomy result. The baseline uses privileged gate positions and can pitch forward without preserving visual gate centering.

Latest editor audit on 2026-06-01:

- The original spin was a real `FPV_HANDOFF_FAILURE`: the solver ran, but
  `frame_fresh` was always `0`, no FPV frames were saved, and the controller
  stayed in its designed yaw-search fallback.
- The root cause was an Elodin version mismatch. The CLI/editor was `0.17.3`
  while the Python package was pinned to `elodin==0.17.2`. In `0.17.3`,
  sensor cameras publish frames automatically to the DB at their configured
  `fps`; the old manual `render_cameras()` path no longer applies.
- After upgrading the sibling Elodin harness to `elodin==0.17.3` and sampling
  `ctx.read_msg("drone.fpv", timestamp=ctx.timestamp)`, a 5 s editor run
  produced `151` FPV frames against a target of about `149`.
- The next failure became `CV_DETECTION_FAILURE`: FPV frames reached the
  solver, but the detector stayed in search because the rendered practice
  gates are dark saturated blue, not the older orange/yellow demo color.
- After adding the Elodin blue HSV range, offline inspection of saved editor
  frames improved from `0/103` usable gate frames to `44/103`, while the
  synthetic orange demo remained `5/5`.
- A subsequent live run proved the Betaflight pitch sign was inverted:
  with detected gates, the drone moved toward negative X while gate 0 is at
  positive X. The Betaflight adapter now maps internal negative pitch-rate to
  RC pitch above center.
- Remaining open issue: repeated editor runs can still produce a bad SITL
  lifecycle where Betaflight exits or the drone remains disarmed before the
  solver trace begins. Treat that as a harness lifecycle problem, separate
  from the camera handoff, detector color, and pitch-sign bugs above.

Follow-up validation on 2026-06-02:

- Fast local checks passed:
  - Python compile for `algorithm/*.py` and `scripts/*.py`
  - orange synthetic gate detection
  - Elodin-like blue synthetic gate detection
  - internal forward pitch maps to Betaflight RC pitch above center
- The sibling Elodin harness still passes `uv run pytest`: `36 passed`.
- The deterministic no-FPV smoke with `RACE_SOLVER=solver.cs260c_pilot`
  passed with successful takeoff and `498` warmup responses.
- A clean 4 s editor run validated the live FPV path:
  - Elodin CLI and Python package both reported `0.17.3+3a11ade`
  - SITL warmup reached `481` responses
  - FPV produced `121` frames against a target of about `119`
  - pilot trace contained `detected=56`, `tracked=27`, `search=40`
  - drone motion moved toward positive X after the pitch-sign fix
    (`x=+1.96m` by 3 s), rather than backward toward negative X
- The 4 s run still did not pass gate 0. Trace overlays showed the gate
  climbing off the top edge of the FPV image: first detections were around
  `pixel_y=29.5`, and the last tracked estimate was around `pixel_y=2.0`.
  At this stage, the bottleneck was vertical/FOV-aware approach control.
- Controller follow-up after that audit:
  - suppress forward pitch when vertical bearing is large
  - increase climb thrust authority
  - keep recent tracked estimates for `0.85 s`
  - let tracked estimates inside that age window command climb even after
    confidence decays
- Offline replay on the same 73 saved editor frames improved from `31/73`
  usable gate frames before the FOV-aware change to `46/73` after it. The
  late top-edge tracked command now climbs with high throttle instead of
  dropping immediately into yaw-search.
- Invalid editor starts can still occur. If the editor log shows Betaflight
  killed, no `logs/elodin_pilot_trace_editor.csv`, and no saved FPV frames,
  discard that run and rerun after cleanup; it is not an algorithm result.
- Camera-sign audit on 2026-06-02 found that Elodin's `sensor_camera.rot_offset`
  pitch sign was opposite our original interpretation. AGP's `20` degree upward
  camera tilt is now represented in the sibling Elodin harness as
  `ELODIN_ROT_OFFSET_PITCH_DEG = -CAM_TILT_UP_DEG`. Details live in
  `docs/SIGN_CONVENTION_AUDIT.md`; run `scripts/audit_sign_conventions.py` to
  repeat the static checks.
- The same audit found a controller-side vertical sign leak: raw
  `bearing_v_rad` is camera-relative, not body-relative. The reactive
  controller now adds the `20` degree camera tilt before using vertical bearing
  for thrust, so a first gate below the optical center can still command climb
  when it is physically above the drone.
- Follow-up completion-first fixes now use the visible outer gate width
  (`2.7m`) for range, map normalized thrust across the full Betaflight throttle
  range, and suppress forward pitch while body-elevation error is large. A
  valid editor run passed gate 0 at `t=6.78s` with position
  `(10.00, -0.11, 1.44)`.
- A temporary RC roll/yaw inversion was tested and rejected because it worsened
  lateral drift. The current remaining harness issue is next-gate reacquisition
  and lateral control after the first gate, not a confirmed horizontal sign
  inversion.
- Additional post-pass tracker hacks were tested and rejected. Filtering for
  only far detections after gate 0 still missed gate 1, and a synthetic
  straight-ahead recovery estimate made lateral drift worse. Keep these as
  negative results: the next useful iteration should redesign lateral
  control/reacquisition with a controlled test, not hide stale old-gate blobs
  behind a tracker-only heuristic.
- `solver/cs260c_pilot.py` now writes `last_gate_passed` and
  `next_gate_index` into the trace so partial-course progress can be separated
  from detector/controller mode changes.
- A fresh 10 s editor validation after backing out the rejected tracker hacks
  produced `303` FPV frames against a target of about `299`, passed gate 0 at
  `t=6.77s`, and ended `1/3` with trace progress
  `latest_last_gate_passed=0`, `latest_next_gate_index=1`. The current verdict
  from `scripts/audit_elodin_editor_run.py` is `PARTIAL_COURSE_PROGRESS`.
- The next controlled audit found that the post-gate trace was not a yaw/roll
  sign problem: yaw and roll were aligned with horizontal bearing, but post-gate
  pitch was exactly zero. A broad attempt to loosen vertical forward
  suppression globally was rejected because it missed gate 0 low
  (`latest_last_gate_passed=-1`). A full attitude-corrected thrust attempt was
  also rejected because it reduced first-gate climb too much.
- The accepted controller change uses attitude/orientation only for forward
  suppression while keeping climb thrust body/camera-relative. This preserves
  first-gate climb but allows forward authority when the gate is centered in
  the tilted FPV image while the drone is pitched down.
- Latest 12 s editor validation completed the local easy course:
  `gates_passed=3/3`, lap time `7.55s`, pass times `[4.90, 6.30, 7.55]`,
  `363` FPV frames against a target of about `359`, and
  `scripts/audit_elodin_editor_run.py` verdict `COURSE_COMPLETE`.

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

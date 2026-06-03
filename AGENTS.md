# CS 260C Project - Drone Gate Racing

## What This Is

This repo is now organized around the current VQ1/final-project algorithm:

```text
FPV gate recognition
  -> temporal gate tracking
  -> reactive navigation state
  -> body-rate/thrust command
  -> simulator/runtime adapter
```

The old PyBullet BC/DAgger/PPO project is preserved under `legacy/pybullet/`
for historical context only. Do not use it as the active project direction or
as the main final-project claim.

## Active System Architecture

```text
Simulator or VQ1 runtime
  -> FPV frame logger + telemetry adapter
  -> GateDetector
  -> GateTracker
  -> ReactiveGateController
  -> RacingCommand
  -> backend adapter
      -> Elodin Betaflight RC today
      -> MAVSDK attitude-rate/thrust for VQ1
```

## Inputs

- FPV camera frames.
- Attitude/orientation telemetry.
- Angular rates, linear velocity, and IMU signals when reliable.

Do not pass GPS, global pose, simulator gate IDs, or pre-known gate coordinates
into `AutonomousRacingPilot`.

World pose from Elodin is allowed only for offline scoring/debugging.

## Output Boundary

The competition-facing algorithm outputs:

```text
RacingCommand(
    roll_rate,
    pitch_rate,
    yaw_rate,
    thrust,
)
```

The Elodin practice harness maps this to Betaflight-style RC fields. The
official VQ1 adapter should map the same command boundary to MAVSDK
attitude-rate/thrust commands.

## Computer Vision

Current CV is deliberately classical and inspectable:

- `algorithm/gate_detector.py`: OpenCV HSV masks, contour extraction, candidate
  scoring, image bearing, and approximate range.
- `algorithm/gate_tracker.py`: confidence filtering, short-memory tracking,
  and `search` / `detected` / `tracked` / `commit` mode assignment.
- `scripts/inspect_gate_frames.py`: offline overlays, masks, and trace output.

The detector defaults should stay aligned with the latest known VQ camera model:
`640 x 360`, intrinsics `[fx, fy, cx, cy] = [320, 320, 320, 180]`, and the
camera/body frame convention documented in `docs/COMPETITION_CONTEXT.md`.

## Navigation And Control

Current navigation is a visual-servo controller, not a learned policy:

- `algorithm/reactive_controller.py`: converts gate bearing/range/confidence
  into body-rate/thrust commands.
- `algorithm/autopilot.py`: owns the detector/tracker/controller loop and logs
  decisions.
- `algorithm/control_adapter.py`: holds the `RacingCommand` abstraction and
  adapter helpers.

Control modes:

- `search`: hover/settle first, then slow yaw scan with no roll or pitch command.
- `detected`: current-frame gate target.
- `tracked`: short-memory target during brief detector drops.
- `commit`: continue through a near gate with damped lateral/yaw correction
  instead of chasing clipped edge/corner detections.
- `recover`: reserved for future safety behavior.

Keep sign/frame checks centralized. Camera pitch, body attitude, ENU/NED, and
RC/MAVSDK mappings are high-risk bug sources.

## Elodin Harness

The local Elodin practice harness lives outside this repo:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix
```

The reproducible sibling-repo changes are captured in:

```text
patches/elodin-ai-grand-prix-cs260c.patch
```

Useful commands:

```bash
scripts/run_elodin_editor.sh
scripts/run_elodin_course_suite.py --courses easy,lateral_soft,low_high,four_gate_straight
```

Latest local validation, 2026-06-02:

```text
easy: COMPLETE gates=3/3 lap_time=7.47
lateral_soft: COMPLETE gates=3/3 lap_time=7.48
low_high: COMPLETE gates=3/3 lap_time=7.42
four_gate_straight: COMPLETE gates=4/4 lap_time=8.69
```

## Validation Commands

Use the project Conda Python:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m py_compile algorithm/*.py scripts/*.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sign_conventions.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_lateral_reacquisition.py
scripts/run_elodin_course_suite.py --courses easy,lateral_soft,low_high,four_gate_straight
```

For the sibling Elodin repo:

```bash
cd /Users/matthewhutchinson/dev/elodin-ai-grand-prix
uv run python -m py_compile sim/course.py sim/main.py solver/cs260c_pilot.py
uv run pytest tests/test_camera_shape.py
```

## Project Rules

- Keep active VQ1/Elodin work out of `legacy/pybullet/`.
- Do not present old PyBullet track results as the current autonomous racing
  algorithm.
- Do not let privileged simulator state leak into the competition-facing
  algorithm.
- Prefer small, inspectable changes and validate with traces/logs before making
  the controller more aggressive.

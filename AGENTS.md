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

The old PyBullet BC/DAgger/PPO project has been removed from the active repo.
Do not use it as the active project direction or as the main final-project
claim.

## Active System Architecture

```text
Simulator or VQ1 runtime
  -> FPV frame logger + telemetry adapter
  -> GateDetector or neural detector backend
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

Current validated CV is deliberately classical and inspectable, with an
optional learned-detector backend for the next perception upgrade:

- `algorithm/gate_detector.py`: OpenCV HSV masks, contour extraction, candidate
  scoring, image bearing, and approximate range.
- `algorithm/neural_gate_detector.py`: optional OpenCV-DNN/ONNX adapter for
  learned detector outputs such as corners, boxes, center/range, or heatmaps.
- `algorithm/gate_tracker.py`: confidence filtering, short-memory tracking,
  and `search` / `detected` / `tracked` / `commit` mode assignment.
- `scripts/inspect_gate_frames.py`: offline overlays, masks, and trace output.
- `external/gatenet/`: ignored upstream GateNet reference checkout. It is not
  imported directly by the active runtime.

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

- `search`: hover/settle, level, brake body velocity when available, then slow
  yaw scan only once level and settled.
- `detected`: current-frame gate target.
- `tracked`: short-memory target during brief detector drops.
- `commit`: continue through a near gate with damped lateral/yaw correction
  instead of chasing clipped edge/corner detections.
- `recover`: reserved for future safety behavior.

Keep sign/frame checks centralized. Camera pitch, body attitude, ENU/NED, and
RC/MAVSDK mappings are high-risk bug sources.

## Learning Scaffold

The active learning path lives under:

```text
learning/
```

The first policy is feature-based:

```text
classical CV gate features + telemetry + tracker history
  -> GRU/MLP policy
  -> RacingCommand
```

Do not train a raw-image CNN first unless official simulator frames show that
classical CV is the bottleneck. Do not use privileged world pose, simulator gate
IDs, GPS, depth, or pre-known gate coordinates as policy inputs.

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

Latest local validation after search level/brake fix, 2026-06-03:

```text
easy: COMPLETE gates=3/3 lap_time=7.57
lateral_soft: COMPLETE gates=3/3 lap_time=7.61
low_high: COMPLETE gates=3/3 lap_time=7.36
four_gate_straight: COMPLETE gates=4/4 lap_time=8.54
circular: DNF gates=2/4 lap_time=16.00
```

Challenge tracks:

```bash
scripts/run_elodin_challenge_suite.sh
```

This runs `circular`/`circular_arc` and `s_curve`, which are intended to expose
the current lack of lookahead/trajectory planning.

## Validation Commands

Use the project Conda Python:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m py_compile algorithm/*.py scripts/*.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sign_conventions.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sequence_selection.py
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

- Do not present old PyBullet track results as the current autonomous racing
  algorithm.
- Do not let privileged simulator state leak into the competition-facing
  algorithm.
- Prefer small, inspectable changes and validate with traces/logs before making
  the controller more aggressive.

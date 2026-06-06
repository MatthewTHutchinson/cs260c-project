# Project Direction Audit - 2026-06-05

## Verdict

The project is moving in the right direction, but the learned controller is not
yet evidence of a race-ready autonomy stack.

The reset was the right move. The active repo now has a clear competition-facing
boundary:

```text
FPV frame + legal telemetry
  -> gate detector / tracker
  -> reactive or learned controller
  -> RacingCommand(roll_rate, pitch_rate, yaw_rate, thrust)
  -> runtime adapter
```

The biggest remaining risk is not file organization or old code mixing back in.
It is trusting offline learning metrics before the controller has closed-loop
success on lateral, circular, and S-shaped recovery cases.

## What Is Going Right

- The old PyBullet BC/DAgger/PPO project is no longer the active project
  direction. Remaining references are warnings or historical context, not active
  runtime code.
- `AutonomousRacingPilot` consumes FPV frames and telemetry and outputs
  `RacingCommand`; it does not receive global pose, GPS, simulator gate IDs, or
  gate coordinates.
- The Elodin practice harness is reproducible through
  `patches/elodin-ai-grand-prix-cs260c.patch`, and the patch still applies
  cleanly to a fresh sibling worktree.
- The active debug course geometry now matches the sibling Elodin harness course
  constants. This removes one obvious source of "we thought the track was X but
  it was actually Y" drift for the current practice suite.
- Sign conventions are now explicitly audited. The current camera tilt,
  vertical bearing, pitch-to-RC, search-hover, roll, and yaw checks pass.
- Sequence/tracker behavior is explicitly tested for common failure cases:
  choosing the next gate when multiple candidates appear, not immediately
  chasing a recently passed gate, and reacquiring after detector loss.
- The learning path has the right shape for the current time budget:
  classical CV features plus telemetry and tracker history into a GRU/MLP policy,
  with `.npz` export for a torch-free runtime.
- Privileged teacher data is correctly framed as an offline label-generation
  tool, not as legal policy input.

## What Went Wrong Or Was Unexpected

- The initial Elodin spin was not navigation behavior. It was an FPV handoff
  failure: no fresh frames reached the pilot, so the controller stayed in search
  yaw.
- The camera pitch sign was wrong at first. The harness needed the camera
  pitched upward by 20 degrees, not effectively downward.
- The visible gate-center issue was real. The early proportional controller
  could chase clipped/edge detections and looked like it was targeting a lower
  gate corner instead of a stable centerline.
- The search behavior needed a reset. Search should hover, settle, and brake
  body velocity before yawing; otherwise a pitched moving drone yaws around a
  bad flight state.
- Offline BC looked better than closed-loop behavior. That was expected in
  hindsight, but still important: low validation loss does not prove that the
  policy can recover after it starts to drift.
- The first learned/relabel closed-loop run still failed badly. On the latest
  audited `easy` learned trace, final lateral error reached 13.82 m. That means
  the learned controller has not yet learned robust lateral/yaw/altitude recovery.
- Some synthetic teacher tracks are already saturating roll/yaw commands,
  especially S-curve variants. That may be useful stress coverage, but it also
  means the teacher is pushing against actuator limits and should be validated
  before we treat it as an expert.

## Fundamental Audit Findings

### 1. Legacy Track Results Are Not Trustworthy Evidence

The old PyBullet project should stay out of the final project claim. Prior audit
notes flagged exactly the kind of benchmark problems we were worried about:
mixed gate counts inside a generalization config, duplicate track names acting
like hidden weighting, and held-out tracks that were too close to training-like
families.

Current status: the active repo is no longer organized around those old tracks.
The current course source is `algorithm/course_library.py`, and it is explicitly
for privileged teacher/debug geometry only.

### 2. Current Debug Tracks Are Aligned, But Not Official Truth

`scripts/audit_course_alignment.py` compares the active course library against
the sibling Elodin harness. It currently passes for:

```text
circular_arc, easy, four_gate_straight, lateral_soft, low_high, s_curve
```

This means the current Mac practice harness and active repo agree. It does not
mean these tracks match VQ1. They are validation surfaces for controller
behavior.

### 3. No Obvious Privileged-State Leak In Runtime Pilot

The runtime boundary is still clean:

```text
AutonomousRacingPilot(frame, telemetry) -> RacingCommand
```

World pose appears in Elodin logs and relabeling tools, but the pilot itself
does not consume it.

One caution: `last_gate_passed` and `next_gate_index` are in the learned feature
vector. At runtime they come from the tracker sequence belief, not simulator
truth. In synthetic/teacher data they can be perfect. That creates a possible
train/test mismatch. Before serious training, run an ablation without these two
features or inject tracker-like noise/dropout so the GRU does not overfit to
perfect gate sequence labels.

### 4. Sign Conventions Look Better, But Closed-Loop Recovery Is Still Broken

The sign audit passes, including:

- 20 degree upward camera tilt
- vertical bearing sign
- internal pitch to RC pitch mapping
- right-gate roll/yaw direction
- hover/settle search behavior

However, the learned closed-loop sign audit still shows the vehicle drifting far
off line. The issue is not simply "roll sign inverted." It is more likely a
combination of weak teacher recovery labels, insufficient closed-loop data, yaw
alignment, and altitude/thrust behavior once the policy leaves the narrow
training corridor.

### 5. Official Simulator Integration Is Still A Major Gap

Elodin is useful for Mac/Linux development, debugging, visualization, and data
generation. It is not the official VQ1 runtime:

- Elodin harness uses Betaflight-style UDP/RC today.
- VQ1 expects MAVLink 2 / MAVSDK-compatible control.
- Elodin world frame is ENU; VQ1 messaging may use NED conventions.
- Elodin can expose world pose for debugging; the official deployed algorithm
  cannot depend on that.

The Windows HP / official simulator path is still necessary as soon as the
simulator is available.

## Audit Commands Run

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m py_compile algorithm/*.py learning/*.py scripts/*.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_course_alignment.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sign_conventions.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_sequence_selection.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_lateral_reacquisition.py
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_privileged_teacher_dataset.py --trace logs/privileged_teacher/trace_augmented.csv --out-dir logs/privileged_teacher/audit_augmented
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python scripts/audit_closed_loop_signs.py --trace logs/elodin_learned_relabel_round1_safe/vq1_pinhole/easy/trace.csv --course easy
```

Sibling Elodin checks:

```bash
git apply --check patches/elodin-ai-grand-prix-cs260c.patch
uv run python -m py_compile solver/cs260c_pilot.py sim/course.py sim/main.py
```

All structural/sign/course-alignment checks passed. The closed-loop learned
trace audit did not pass as a behavior claim; it exposed the current failure.

## Direction From Here

1. Keep the current architecture.
   The FPV/tracker/controller/command boundary is the correct project spine.

2. Do not train seriously from the old reactive traces.
   Use them only as smoke tests.

3. Improve the teacher before spending T4 time.
   The teacher needs stronger lateral recovery, yaw alignment, altitude hold,
   and near-gate commit behavior.

4. Treat circular and S-curve tracks as failure-finding tracks.
   Do not tune only on `easy`.

5. Add a learned-policy ablation without perfect sequence features.
   The policy should work from visual/tracker evidence, not hidden knowledge of
   true next gate IDs.

6. Use the T4 only after the teacher data closes the loop locally.
   GPU training is cheap compared with debugging bad labels.

7. Bring up the official simulator on Windows as soon as possible.
   That is the only way to verify MAVSDK control, camera timing, telemetry
   fields, and the real VQ1 runtime assumptions.

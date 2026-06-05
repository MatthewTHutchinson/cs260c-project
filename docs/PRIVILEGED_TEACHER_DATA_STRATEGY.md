# Privileged Teacher Data Strategy

Date: 2026-06-05

## Decision

Use privileged simulator information to generate high-quality expert labels, but
do not feed privileged information to the deployed policy.

This is the core split:

```text
teacher / data generation:
  may use simulator pose, gate positions, gate IDs, and debug course geometry

student / competition-facing policy:
  may use only FPV-derived features and allowed telemetry
```

The goal is not to make the final system depend on simulator truth. The goal is
to use simulator truth to produce better examples than the current reactive
controller can provide.

## Why This Is Useful

The current CSV traces are weak demonstrations:

- the controller sometimes targets clipped gate edges or corners
- lateral displacement is poorly handled
- curved and S-shaped tracks expose missing lookahead
- behavioral cloning would copy those failures

Privileged teacher data fixes the label quality problem. A teacher can know the
true course geometry, generate a fast smooth line, and label what the command
should have been from each observed state.

## Training Pattern

Use learning from privileged information:

```text
privileged expert:
  world pose + gate positions + gate order
  -> minimum-snap / racing-line / lookahead trajectory
  -> expert local target or RacingCommand label

student policy:
  classical CV features + tracker history + telemetry
  -> GRU/MLP policy
  -> RacingCommand
```

The privileged fields can be logged for teacher construction, but they must be
excluded from the student feature vector.

## First Teacher To Build

Start with a simple sequence-aware expert before a full aggressive racing
planner:

1. read true drone pose and true gate centers/normals from the local simulator
2. identify the next gate and one lookahead gate
3. construct a local corridor target through the center of the next gate
4. generate a smooth reference segment toward that corridor
5. convert reference tracking into `RacingCommand` labels
6. log both the privileged teacher fields and the legal student features

Initial scaffold:

```bash
python scripts/generate_privileged_teacher_dataset.py \
  --out logs/privileged_teacher/trace.csv
```

This produces a debug-course teacher CSV with legal student features,
`teacher_*` command targets, and auditable `world_*` / `teacher_next_gate_*`
privileged columns that are excluded from the current learning feature vector.

Once that works, upgrade the reference generator:

```text
gate centers/normals
  -> minimum-snap trajectory
  -> time allocation
  -> desired velocity/acceleration/yaw
  -> body-rate/thrust label or local target label
```

## Dataset Rows

Each row should keep the split explicit.

Legal student inputs:

```text
timestamp_s
FPV-derived bearing/range/confidence
tracker mode
recent feature history
attitude/orientation
angular rates
linear velocity
previous command
```

Teacher-only/debug fields:

```text
world pose
world velocity if privileged
true next gate ID
true gate center
true gate normal
distance to gate plane
minimum-snap reference state
```

Targets:

```text
teacher_roll_rate_rad_s
teacher_pitch_rate_rad_s
teacher_yaw_rate_rad_s
teacher_thrust_norm
```

The training loader should consume the legal fields plus targets. It should not
consume the teacher-only/debug fields.

## Simulator Placement

Use the Mac/local simulator for teacher-data generation if it is already running
and can expose debug pose/gate geometry.

Use the Windows HP for official simulator bring-up and logging. If the official
simulator exposes privileged fields, they are debug/teacher-only and must stay
out of the deployed policy.

Use the T4 for training once useful labels exist. Pause the T4 while we are
building the teacher/data-generation path, because the GPU is not the bottleneck
until there is a better dataset.

## Elodin On The T4

Elodin supports macOS/Linux in principle, but the T4 VM is a headless cloud
training box. Running the full local simulator there may require:

- Elodin CLI and Python package setup
- Betaflight SITL build
- render-server/editor lifecycle
- graphics/display forwarding or headless rendering validation
- repository patch synchronization with the sibling harness
- process cleanup and logging wrappers

That setup may be worth doing later if we want cloud rollouts. It is not the
highest-leverage next step. The near-term split is:

```text
Mac / Windows:
  run simulators and generate labeled rollouts

T4:
  train and evaluate BC models from those logs
```

## Success Criteria

Before serious BC training, produce a dataset where:

- the teacher flies through gate centers, not clipped corners
- lateral offsets are corrected before gate pass
- circular and S-shaped tracks have usable lookahead labels
- student inputs remain legal
- a held-out trace can be trained/evaluated by `learning/train_bc.py`

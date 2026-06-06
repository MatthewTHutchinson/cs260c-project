# Next Steps: VQ1 Setup And Learned Policy Roadmap

Date: 2026-06-04

This project should move on three tracks in parallel:

1. get the official Windows simulator running and logging real VQ1-style data
2. improve navigation with sequence-aware lookahead and minimum-snap references
3. build a learned-policy path from the current inspectable controller

The Windows machine is worth picking up. It should not derail the current
algorithm work, but it is the only way to validate the official simulator zip,
camera stream, telemetry, command interface, and runtime assumptions.

## Decision

Use the HP for official simulator bring-up and data capture, not as the main
training machine.

The Mac/local simulation environment remains the fast development loop for
controller changes, dataset generation, learned-policy experiments, and cloud
training preparation. The HP is for compatibility checks:

- does the official simulator launch?
- what exact UDP/MAVSDK connection does it expose?
- do camera frames and telemetry match the published spec?
- can we stream body-rate/thrust commands below the command-rate limit?
- can we log frames, telemetry, commands, and result summaries?

## Immediate Next Steps

### 1. Pick up and prepare the HP

Do this first after pickup:

- create the Windows account
- install or verify:
  - Python 3.10 or 3.11
  - Git
  - VS Code
  - NVIDIA driver status, if the HP has an NVIDIA GPU
  - simulator zip extraction path with no spaces if possible
- copy or clone this repo
- unzip the AI-GP simulator package
- run the simulator once manually before attempting autonomy

The first win is not racing. The first win is confirming that the official
runtime opens, renders frames, exposes telemetry, and accepts commands.

### 2. Build the VQ1 adapter boundary

Keep the competition-facing boundary fixed:

```text
AutonomousRacingPilot
  -> RacingCommand(
       roll_rate,
       pitch_rate,
       yaw_rate,
       thrust,
     )
  -> MAVSDK / official simulator adapter
```

Do not pass GPS, global pose, simulator gate IDs, or known gate coordinates into
the pilot. If the official simulator exposes any privileged state, treat it as
debug-only.

### 3. Log the official simulator

Before making the controller smarter, collect synchronized logs:

```text
timestamp
FPV frame path
attitude / orientation
angular rates
linear velocity if available
system flags
RacingCommand output
detector/tracker mode
gate observation
run result
```

These logs become the bridge from the current controller to learned policies.

### 4. Use the GCP T4 for training runs

The available GCP T4 and cloud credits are enough for meaningful experiments if
the experiments stay compact. Use the cloud machine for:

- behavioral-cloning training runs
- DAgger dataset aggregation and retraining
- compact PPO fine-tuning
- robustness sweeps over randomization/noise settings

Do not spend the credits debugging environment setup. Smoke-test each script
locally first with a tiny dataset or one short rollout, then run the longer job
on the T4.

## Learned Policy Roadmap

The long-term learned system is:

```text
FPV frame sequence
  -> compact CNN / neural gate detector
  -> gate features or image embedding
  -> GRU/LSTM temporal policy with telemetry
  -> MLP action head
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

The important design choice is that the learned controller uses the same output
boundary as the current controller. That lets us swap the controller without
rewriting the runtime adapter.

The near-term version should not start with raw-image CNN control. The current
classical CV system already produces useful gate features:

```text
gate center
bearing
rough range
confidence
tracker mode
recent target history
```

Use those as the first learned-policy input. Add a CNN only when image-level
information is clearly the bottleneck.

Near-term learned system:

```text
classical CV gate features + telemetry + tracker history
  -> GRU/MLP temporal policy
  -> roll_rate, pitch_rate, yaw_rate, thrust
```

This is much cheaper to train, easier to debug, and better aligned with the
time budget.

## Minimum-Snap And Lookahead

Minimum-snap trajectories are useful, but not as the final competition-facing
input if they require privileged global gate coordinates. Use them in two safer
ways:

1. offline teacher/reference generation in simulation
2. local lookahead targets built from perceived gate sequence estimates

The practical pipeline is:

```text
detected gate sequence
  -> local corridor or next-gate target
  -> minimum-snap / smooth reference segment
  -> body-rate/thrust tracking command
```

Minimum-snap helps with curved tracks because it gives the controller a smooth
line through a sequence instead of reacting only to the currently visible gate.
The key is to keep the competition-facing version perception-based: it can use
estimated local gate geometry, not hidden simulator ground truth.

Use minimum-snap first as a teacher:

```text
known debug course geometry
  -> generate smooth expert trajectory
  -> label commands or local targets
  -> train BC/DAgger policy
  -> deploy policy using only FPV-derived features
```

This gives the learned policy better behavior than cloning the current reactive
controller alone, especially for circular and S-shaped tracks.

See `docs/PRIVILEGED_TEACHER_DATA_STRATEGY.md` for the explicit training split:
privileged simulator truth is allowed for teacher/data generation, while the
student policy consumes only FPV-derived features and allowed telemetry.

## Architecture

Start compact. There are two architecture levels.

Level 1, recommended first:

```text
feature encoder:
  MLP over detector/tracker outputs, attitude, angular rates, velocity

temporal model:
  GRU, hidden size 64-128

action head:
  MLP -> roll_rate, pitch_rate, yaw_rate, thrust
```

Level 2, only after feature-policy experiments:

```text
image encoder:
  small CNN, MobileNetV3-small, EfficientNet-lite, or ResNet-18

telemetry encoder:
  MLP over attitude, angular rates, linear velocity, and status flags

temporal model:
  GRU or LSTM, hidden size 128-256

action head:
  MLP -> tanh-normalized roll_rate, pitch_rate, yaw_rate
       -> sigmoid or clipped thrust
```

Use GRU first. It is simpler than LSTM and usually enough for short-horizon
memory through gate occlusion, fast yaw turns, and brief detector failure.

## CNN Decision

Do not implement the CNN first unless the official simulator frames make
classical CV unreliable.

Use classical CV first because:

- it already works on simple tracks
- it is inspectable during failures
- it creates low-dimensional features that train cheaply on a T4
- it lets BC/DAgger/PPO focus on navigation and control instead of perception
- it avoids spending cloud credits on a labeling/data problem

Add CNN perception when one of these becomes true:

- gate colors or lighting vary enough to break the HSV/contour detector
- distractors produce too many false gates
- range/orientation estimates are too noisy for lookahead
- official simulator visuals differ substantially from the local environment
- you have enough logged frames to train and validate a detector offline

## Training Order

### Stage A: Behavioral cloning

Use two teachers:

1. the current reactive controller only for smoke tests and simple safe behavior
2. a minimum-snap/lookahead expert for smoother curved-track behavior

Do not treat the current local trace CSVs as high-quality demonstrations. They
are useful for proving the T4 training plumbing, but the teacher still has known
failure modes: edge/corner chasing, weak lateral displacement handling, and poor
turn behavior on circular or S-shaped tracks.

Input:

```text
classical CV features + tracker history + telemetry
```

Target:

```text
teacher RacingCommand or local target command
```

Goal:

- learn a stable imitation of a better teacher
- prove the GRU policy can run in the loop
- do not optimize speed yet

### Stage B: DAgger

Run the learned policy in simulation, but keep the teacher available.

For each rollout:

1. learned policy acts
2. teacher labels what it would have done from the visited states
3. aggregate those states into the dataset
4. retrain the policy

DAgger matters because pure behavioral cloning only sees states the teacher
visits. Once the learned policy drifts, it needs labels for off-nominal states.

### Stage C: PPO fine-tuning

Only do PPO after the BC/DAgger policy can complete simple courses.

Reward should prioritize:

- valid gate sequence progress
- completion
- staying level enough to preserve visual tracking
- keeping the next gate in view
- smooth commands
- avoiding excessive yaw/spin/search behavior

PPO is for improving robustness and speed, not for discovering flight from
scratch.

## T4 Experiment Plan

Use the T4 in short, staged jobs:

1. dataset sanity check: train on a tiny log and overfit it
2. current-trace smoke check: verify real CSV columns load and train
3. teacher upgrade: build sequence-aware/lookahead or minimum-snap labels
4. BC baseline: train GRU/MLP on upgraded teacher logs
5. BC evaluation: run policy on easy/lateral/height/four-gate tracks
6. DAgger: collect failure states and relabel with teacher
7. PPO: fine-tune only after BC/DAgger completes simple tracks
8. randomization sweep: evaluate lighting/noise/latency/camera perturbations

Expected cloud use should stay modest if the first policy is feature-based
instead of raw-image-based.

Use `learning.eval_policy` for BC evaluation. It now reports overall,
per-course, and per-mode errors, plus prediction-vs-target saturation
percentages so curved-track failures are visible instead of hidden in aggregate
MSE.

Use `--exclude-courses` during training and `--include-courses` during
evaluation for leave-one-course-out checks. The current leave-`s_curve`-out
result fails badly on the base dataset, but improves after adding randomized
curved and S-shaped teacher courses. That makes dataset diversity the next
primary lever before simply scaling the model.

The trained GRU can now be loaded through
`algorithm.learned_controller.LearnedFeatureController` and attached to
`AutonomousRacingPilot` as an optional controller. Use
`scripts/smoke_learned_controller.py` before any simulator run to confirm the
checkpoint loads, normalizes features, preserves sequence history, and emits a
clipped `RacingCommand`.

Prefer checkpoints trained with `--no-prev-command-features` for the first
runtime rollouts. Previous-command features improve teacher-forced offline MSE,
but without DAgger/closed-loop data they create a train/runtime mismatch because
the deployed policy feeds back its own previous predictions.

Also run a stricter checkpoint with `--no-sequence-features`. This drops
`last_gate_passed` and `next_gate_index`, which are legal only if they come from
the tracker sequence belief at runtime but can be unrealistically perfect in
privileged teacher data. Use `scripts/audit_learning_feature_spec.py` as the
guard before T4 runs. Regenerate old teacher traces first if they predate the
`frame_fresh` column; missing selected features should fail the audit, not turn
into silent zeros.

```bash
python scripts/audit_learning_feature_spec.py \
  --no-prev-command-features \
  --no-sequence-features \
  --expect-no-prev-command-features \
  --expect-no-sequence-features \
  --trace logs/privileged_teacher/trace_augmented.csv
```

Use `scripts/compare_controllers_on_trace.py` as the last offline gate before
simulator rollout. On the current S-curve trace, the no-prev-command learned
checkpoint is much closer to the privileged teacher than the reactive
controller:

```text
learned_vs_teacher mse=0.00116282
reactive_vs_teacher mse=0.17836463
```

The Elodin rollout path is:

```bash
scripts/run_elodin_learned_suite.sh
```

It defaults to the exported `.npz` checkpoint so the sibling Elodin runtime does
not need `torch` installed. It sets `CS260C_LEARNED_CONTROLLER_CHECKPOINT` for
the sibling solver and keeps reactive fallback enabled for search/lost-gate
states.

First closed-loop smoke result, 2026-06-04:

```text
course=easy
sim_time=4.0s
status=DNF
gates_passed=0/3
trace_rows=178
modes={'detected': 178}
```

This is still useful progress: the learned controller is now running in the
simulator loop, but the rollout shows why the current CSVs are smoke tests only.
The policy enters learned mode during the takeoff/transient phase, commands
near-minimum thrust, and accumulates lateral velocity before gate 1. Do not
spend more T4 time scaling the model until the teacher labels include launch
guards, off-nominal lateral recovery, and closed-loop-like deviations from the
centerline.

Follow-up after adding augmented teacher episodes:

```text
trace_augmented.csv rows=14280 courses=26
augmented padded checkpoint=feature_bc_augmented_padded_leave_s_curve_out_20e_no_prev.npz
10 s learned easy rollout: DNF, gates=0/3
first learned thrust: 0.300 -> 0.722
final z after 4 s smoke: 0.46 m -> 1.29 m
10 s rollout reached x=9.92 m near gate 0 but y=1.62 m, so it missed laterally
```

Follow-up after removing both previous-command and sequence-index features,
2026-06-06:

```text
checkpoint=feature_bc_augmented_no_prev_no_seq_leave_s_curve_out_20e.npz
feature_count=22
sequence_features=none
prev_command_features=none
privileged_features=none
heldout_s_curve_mse=0.02086311
heldout_s_curve_mae_yaw_rate=0.12600399
comparison_s_curve learned_vs_teacher mse=0.02080014
comparison_s_curve reactive_vs_teacher mse=0.17126263
phase=off_nominal learned_vs_teacher mse=0.11474959
```

This is a cleaner final-project baseline than the earlier checkpoint because it
does not depend on perfect teacher-side sequence labels or previous-action
feedback. It is weaker than the easiest offline teacher-forced models, but that
is the point: it is a more honest estimate of what the deployable feature policy
can learn. The remaining failure is still recovery behavior, especially roll/yaw
authority during off-nominal and near-gate commit states.

Recovery-weighted training was tested next:

```text
unweighted:
  comparison_s_curve learned_vs_teacher mse=0.02080014
  phase=off_nominal learned_vs_teacher mse=0.11474959
off_nominal=2:
  comparison_s_curve learned_vs_teacher mse=0.02373977
  phase=off_nominal learned_vs_teacher mse=0.13114854
off_nominal=4, commit=2:
  comparison_s_curve learned_vs_teacher mse=0.02709405
  phase=off_nominal learned_vs_teacher mse=0.14666850
```

This is a useful negative result. Simply sampling recovery/commit rows more
often lowers training/validation loss on the source distribution but does not
improve held-out S-curve recovery. The project should move toward better
recovery labels and reference generation rather than trying to tune sampling
weights.

The recovery teacher was then changed from nominal lookahead to a local
path-rejoin target:

```text
recovery_teacher=rejoin
checkpoint=feature_bc_augmented_rejoin_no_prev_no_seq_leave_s_curve_out_20e.npz
feature_count=22
sequence_features=none
prev_command_features=none
privileged_features=none
heldout_s_curve_mse=0.00185871
heldout_s_curve_mae_yaw_rate=0.05335076
comparison_s_curve learned_vs_teacher mse=0.00173593
comparison_s_curve reactive_vs_teacher mse=0.14797705
phase=off_nominal learned_vs_teacher mse=0.00490683
```

This is the strongest offline result so far. The lesson is useful for the final
project: changing the teacher/reference made the feature GRU much more
learnable, while sampling-weight tricks did not. The `rejoin` teacher slows
forward pitch and labels lateral/yaw/altitude recovery toward a near future
point on the reference line before gate commit.

This still does not prove closed-loop racing. It proves the deployable feature
policy can imitate a better recovery reference on a held-out S-curve without
previous-command feedback or perfect sequence labels. Closed-loop rollout is the
next evidence layer when a simulator is available.

Closed-loop rejoin rollout follow-up, 2026-06-06:

```text
checkpoint=feature_bc_augmented_rejoin_no_prev_no_seq_leave_s_curve_out_20e.npz
course=easy

pure learned:
  status=DNF, gates=0/3
  end_lateral_error_m=-12.853730
  modes={'detected': 627}

safety-gated learned with reactive fallback:
  status=DNF, gates=0/3
  end_lateral_error_m=13.080810
  modes={'detected': 553, 'commit': 38, 'tracked': 44}
```

This is the most important current limitation. The safety gate improved the
first few seconds by keeping the policy inside a narrower feature envelope, but
it did not turn offline imitation into closed-loop navigation. The learned
policy still leaves the training corridor, then the perception/control loop
cannot recover in time. Do not spend T4 budget on larger BC runs until the
dataset contains closed-loop failure states and the teacher/recovery reference
can bring those states back to the gate line.

This means the direction is correct but incomplete. The next teacher upgrade is
closed-loop relabeling: log privileged debug world position from failed Elodin
rollouts, compute the desired lookahead/minimum-snap correction from that
off-line state, and train those labels back onto the same legal FPV/telemetry
features.

Closed-loop relabeling is now implemented through:

```bash
scripts/relabel_closed_loop_trace.py
scripts/audit_closed_loop_signs.py
```

The first relabel iteration proved the data loop but did not solve the easy
course. It fit the relabeled failure trace offline (`mse=0.00120523`) but the
8 s simulator rollout still finished `DNF gates=0/3`, low and far right of the
gate line. The audit showed roll correction had the right sign, while yaw and
altitude/thrust recovery still need a better teacher.

The second relabel iteration moved the relabeler to the same path-rejoin idea
used by the synthetic `rejoin` teacher:

```bash
python scripts/relabel_closed_loop_trace.py \
  --trace logs/elodin_learned_rejoin_guard_smoke/vq1_pinhole/easy/trace.csv \
  --course easy \
  --teacher rejoin \
  --episode-id easy:closed_loop:rejoin_guard_001 \
  --out logs/privileged_teacher/closed_loop_relabels/easy_rejoin_guard_001_rejoin.csv
```

```text
rows_relabelled=312
closed_loop_relabel_mse=0.00080114
heldout_s_curve_mse=0.00213601
10s_easy_rollout=DNF gates=0/3 end_lateral_error_m=12.440920
10s_source_logging=DNF gates=0/3 command_sources={'reactive_fallback': 528, 'learned': 131}
6s_source_logging=DNF gates=1/3 gate0_pass_t=5.25
```

This is a small improvement over the previous guarded learned rollout
(`end_lateral_error_m=13.080810`), but still not a completion result. A naive
lateral-velocity recovery rule was also tested and made the miss worse, so it
was not kept. The next local step is to collect several diverse failed rollouts,
relabel them with the rejoin/reference teacher, and use command-source logging
to separate learned-policy failures from fallback/supervisor failures. The
latest 10 s failure was mostly `reactive_fallback`, so the fallback/supervisor
boundary is part of the problem.

The third relabel iteration used three source-logged rejoin relabel episodes
and is the best closed-loop result so far:

```text
checkpoint=feature_bc_augmented_rejoin_plus_3relabels_no_prev_no_seq_20e.npz
heldout_s_curve_mse=0.00191887
all_relabels_mse=0.00100016
10s_easy_rollout=DNF gates=1/3 gate0_pass_t=5.19
command_sources={'reactive_fallback': 442, 'learned': 189}
```

The fourth relabel iteration added the new post-gate failure trace. It improved
offline held-out S-curve MSE (`0.00185186`) but regressed the 10 s easy rollout
to `DNF gates=0/3`. This is a useful warning: DAgger-style relabeling needs
source-aware selection/balancing. Adding a hard post-gate failure episode can
overcorrect the first-gate approach if it is mixed naively with the rest of the
teacher data.

The first source-aware weighting experiment downweighted the hard post-gate
trace to `0.25`. It improved offline S-curve MSE again (`0.00175359`) but still
finished `DNF gates=0/3` with `end_lateral_error_m=-5.138923`. That narrows the
next step: build a supervisor/state split for first-gate approach versus
post-gate reacquisition instead of trying to solve both by scalar sampling
weights inside one blended BC policy.

## Robustness Work

Randomization should be added before trusting learned-policy results:

- gate color and brightness
- lighting exposure
- camera noise and blur
- small camera intrinsics changes
- camera pitch perturbation
- command latency and dropped frames
- motor response lag
- telemetry noise
- initial pose offsets
- wind or lateral drift if supported
- distractor objects and partial gate occlusion

Also keep the state machine. The learned policy should not own every safety
decision at first.

Recommended supervisor:

```text
search:
  hover, level, brake drift, yaw scan

detected/tracked:
  learned policy or visual-servo policy controls gate approach

commit:
  damp lateral corrections and continue through gate

recover:
  reserved for unstable attitude, lost frames, or repeated missed gates
```

This makes the learned policy easier to train because it only has to learn the
most valuable part first: visual gate approach and turning behavior.

## What Not To Do Next

Do not spend the next block of time on:

- training a giant video world model
- implementing CNN perception before verifying classical CV fails
- pure PPO from random actions
- rewriting the whole simulator stack
- depending on old PyBullet track results
- using privileged global pose as a policy input

The fastest credible path is:

```text
official simulator bring-up
  -> synchronized logging
  -> sequence-aware lookahead / minimum-snap teacher
  -> feature-based BC policy from controller and teacher
  -> DAgger for recovery states
  -> PPO for robustness and speed
  -> CNN detector only if classical CV becomes the bottleneck
  -> VQ1 adapter hardening
```

## Research-Aligned Roadmap

The literature points to a hybrid system rather than a one-piece neural network
as the most credible path:

```text
FPV frames + IMU/telemetry
  -> neural or classical gate perception
  -> gate-relative state / tracker / drift correction
  -> lookahead reference or local trajectory
  -> learned or optimized body-rate/thrust controller
  -> safety supervisor and runtime adapter
```

For this project, the practical version is:

1. keep the current reactive controller as the closed-loop baseline
2. add sequence-aware lookahead so curved tracks are not handled gate-by-gate
3. use privileged geometry only offline to generate high-quality teacher labels
4. train the feature GRU/MLP on those labels
5. collect failed closed-loop states and relabel them with the teacher
6. use DAgger-style aggregation until the learned policy completes easy tracks
7. only then spend T4 time on PPO or a CNN perception upgrade

The main blockers are not compute. The main blockers are:

- no official simulator logs yet
- no verified MAVSDK adapter yet
- local simulator/runtime mismatch with the official VQ1 environment
- closed-loop drift after small policy errors
- weak labels for recovery, yaw alignment, and near-gate commit
- possible perception brittleness once lighting/distractors change
- limited time to debug a raw-image CNN or pure PPO setup

In other words: the project should use deep learning, but the next deep-learning
work should be data/relabeling and supervisor-aware imitation first, not a
larger network trained on the same narrow traces.

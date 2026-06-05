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

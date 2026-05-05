# Brainstorming: Next Improvements, Limits, and Transfer Paths

## Why this document exists

This file captures the current brainstorming thread around:

- how to keep improving the drone racing policy step by step
- what the current policy can and cannot realistically become
- how to evolve the observation space toward richer state and computer vision
- when transfer learning is possible versus when major retraining is unavoidable

## Current policy snapshot

The current main policy is a small MLP actor / actor-critic:

- input: `36` floats
- structure: `36 -> 256 -> 256 -> 4` for BC/DAgger
- PPO variant: shared `36 -> 256 -> 256` trunk, plus actor mean head, learned log-std, and critic head
- output: `4` normalized action values in `[-1, 1]`

Current observation contents:

- body-frame velocity `(3)`
- body-frame angular rate `(3)`
- relative position of next gate in body frame `(3)`
- relative position of second-next gate in body frame `(3)`
- stacked across `3` timesteps, so `12 * 3 = 36`

Current action contents:

- `dx_b`
- `dy_b`
- `dz_b`
- `d_yaw`

These actions are not direct motor commands. They are body-frame waypoint deltas and yaw deltas that get converted into a world-frame target for a classical PID controller.

## What the current design is good at

- stable learning from imitation
- strong robustness compared with training PPO from scratch
- good generalization across moderate variations in gate layouts
- clean control abstraction because PPO does not have to discover low-level motor control

## What limits the current ceiling

The current system is strong for a class project and a simulator benchmark, but several design choices limit the ultimate speed ceiling:

- The policy is reactive, not explicitly planning over a long horizon.
- The observation only includes the next two gates, which can be too little context for sharp switchbacks.
- Gate orientation is not explicitly included in the observation.
- The current expert is still a heuristic gate chaser, not a true time-optimal planner.
- The action is a small waypoint delta rather than a richer trajectory or direct thrust command.
- The reward mainly encourages safe progress and gate completion, not true minimum-time racing.
- The state input is privileged simulator information, not camera-based perception.

This means the current policy can probably become quite strong inside the present simulator formulation, but it is not yet close to a full autonomous drone racing stack.

## Best step-by-step improvement ladder

### Step 1: finish and compare the current zigzag-focused PPO run

Immediate goal:

- let `logs/ppo_generalization_balanced_v2` finish
- compare it against:
  - `logs/ppo_generalization_v1/policy_ppo_best.pt`
  - `logs/ppo_generalization_balanced_v1/policy_ppo_best.pt`

Main question:

- did the dedicated zigzag training and validation recover `heldout_zigzag` without losing easier-track speed?

### Step 2: retrain BC and DAgger on the enriched track distribution

Right now PPO is still warm-started from imitation data that was collected before the zigzag-focused training distribution existed.

If zigzag remains weak, the next strong move is:

- rebuild BC data on the enriched track set
- rerun DAgger on that same distribution
- then fine-tune PPO again

This should give PPO a better imitation prior for aggressive switchback behavior.

### Step 3: improve the observation space without going full vision yet

Best low-risk observation upgrades:

- add gate normal / orientation information
- add third-gate lookahead
- add relative heading to the gate plane
- optionally add a small amount of history beyond simple frame stacking

This keeps the problem structured and should be easier than jumping straight to end-to-end vision.

Status update:

- implemented in code as `configs/generalization_obs_v1.yaml`
- current richer state branch uses:
  - `3`-gate lookahead
  - gate normals in the observation
  - explicit `cos/sin` heading alignment to the next gate plane
- resulting obs dim is `78`
- first full training run on this branch produced the current best overall checkpoint:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`

### Step 4: improve the expert

The current expert is enough to bootstrap flight, but it is not the ceiling we want the learner to regularize toward forever.

High-value improvements:

- spline or minimum-snap style trajectory generation
- explicit corner-cutting / racing-line behavior
- speed profile planning through gates

This helps BC, helps DAgger labels, and makes the PPO BC auxiliary loss less conservative.

Status update:

- a first expert upgrade is now implemented
- the expert no longer just chases the next gate center
- it now aims slightly beyond the next gate plane while blending in the next two gate directions and normals
- this expert upgrade, combined with the richer observation branch, materially improved BC, DAgger, and PPO results

### Step 5: make the reward more time-optimal

Current reward mostly says:

- pass gates
- avoid crashing
- move toward the next gate
- avoid jerky control

To race faster, future reward work could include:

- stronger incentive for shorter lap time
- speed-through-gate shaping
- better penalty for unnecessary path length
- less conservative smoothness penalties after stable flight is achieved

### Step 6: add vision in a staged way

Do not jump directly from privileged state to vision-only if it can be avoided.

A better sequence is:

1. state only
2. state + richer geometry
3. state + image
4. image-dominant or image-only if desired

This lets the existing policy act as a teacher while the vision pathway learns.

## Do we need to retrain from scratch when we add more observations?

Short answer: not always.

It depends on what kind of observation change we make.

### Case A: add more scalar / geometric observations

Examples:

- add gate normals
- add third-gate lookahead
- add explicit distance-to-plane
- add more kinematic features

In this case, we do **not** need to start from scratch.

Good transfer strategy:

- increase `obs_dim`
- expand the first layer to accept the new features
- copy the old first-layer weights for the original `36` input dimensions
- initialize the new columns for the added features to zero or small random values
- copy the deeper layers directly
- fine-tune with BC or DAgger first, then PPO

This is the easiest and cleanest transfer path for the current repo.

### Case B: add a new observation branch while keeping the old state input

Examples:

- keep the `36`-D state input
- add a second branch for images
- fuse state features and image features later in the network

In this case, we also do **not** need to start from scratch.

Good transfer strategy:

- keep the current state encoder / trunk
- initialize it from the current policy
- add a vision encoder branch
- initialize the vision encoder from a pretrained model if available
- fuse both branches into a new policy head
- train the fused policy with imitation first
- optionally distill from the old state-only policy while the vision encoder learns

This is the most practical path toward computer vision in this project.

### Case C: switch to vision-only

Examples:

- remove privileged state from the deployed policy
- policy sees only RGB image or image sequence

In this case, direct transfer becomes much weaker.

We still may not be literally training "from zero," but we should expect substantial retraining.

Useful transfer ideas here:

- use a pretrained vision backbone
- use teacher-student distillation from the current state-based policy in simulation
- keep the action head or parts of the control head if dimensions align
- do imitation learning first before PPO

So this is not pure scratch training, but it is much closer to a new model than Case A or B.

## Best recommended transfer-learning path for this repo

If the goal is to evolve toward richer observations and CV while preserving progress, the best path is:

1. keep the current state pathway
2. add richer geometric features first
3. transfer the current MLP/trunk weights
4. retrain with BC/DAgger on the new input format
5. fine-tune with PPO
6. only then add a vision branch
7. use the state policy as a teacher for the multimodal model
8. decide later whether full vision-only deployment is worth the extra complexity

## When full retraining is most justified

Full or near-full retraining is most justified when:

- the observation modality changes completely
- the action space changes substantially
- the control abstraction changes from waypoint deltas to something more expressive
- the task changes from simulator-state racing to camera-based real-world deployment

Even then, imitation, partial weight transfer, and distillation can still reduce how much work "from scratch" really means.

## Practical next experiment ideas

- finish `ppo_generalization_balanced_v2`
- rerun BC and DAgger on the zigzag-enriched distribution if needed
- add gate normals to the observation
- add a third-gate lookahead
- compare `36`-D versus richer-state MLPs before adding images
- prototype a multimodal policy with:
  - current state encoder
  - lightweight image encoder
  - late fusion

## Working conclusion

The current policy still has meaningful headroom inside the simulator, especially through better training distributions, a better expert, richer state observations, and more targeted reward design.

We do not need to throw away the current policy to make progress.

For richer state inputs, transfer learning should be straightforward.
For multimodal state + vision, transfer learning is still very realistic.
For pure vision-only racing, we should expect major retraining, but teacher-student transfer can still help a lot.

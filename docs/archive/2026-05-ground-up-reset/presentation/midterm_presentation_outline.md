# Monday Presentation Outline

Target length: `4.5` minutes

Recommended deck size: `5` slides

Core thesis:

I built a simulator-side autonomous drone racing pipeline motivated by the AI Grand Prix / DCL competition setting. The strongest result came from combining staged imitation-to-RL training with better geometric observations, while harder audits exposed that the current policy is not yet broadly robust or competition-ready.

## What You Need To Make

### Required Assets

- `1` short simulation video:
  use the current richer-observation PPO champion, `10-15` seconds.
- `1` architecture diagram:
  `Vision/State -> Policy -> Local waypoint target -> PID -> Drone`
- `1` training pipeline diagram:
  `Expert -> BC -> DAgger -> PPO`
- `1` small results table:
  compare earlier PPO champion vs richer-observation PPO.
- `1` limitation slide:
  show the extended audit finding, especially mirrored/right-turn and longer-course weakness.

### Best Video Command

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.render_demo_video \
  --preset state_champion \
  --output logs/presentation/state_champion_demo.mp4 \
  --episodes 1 \
  --track-name heldout_diamond \
  --force-visuals \
  --shadow
```

If that render is too slow, use live visualization and screen record:

```bash
/Users/matthewhutchinson/miniconda3/envs/cs260c-project/bin/python -m eval.visualize \
  --preset state_champion \
  --episodes 1
```

## Slide 1: Problem And Motivation

Title:

`Autonomous Drone Racing for an AI Grand Prix-Style Competition`

Main message:

This project is motivated by a competition setting where a drone must autonomously navigate gates using perception and telemetry, not manual control.

Say:

`The motivation is the AI Grand Prix / DCL-style autonomous drone racing problem. The latest spec I used describes a forward camera stream, MAVLink telemetry and control, deterministic simulation, and no GPS or easy global-position shortcut. My project focuses on the learning pipeline for that problem inside a PyBullet drone racing simulator.`

Show:

- simulator screenshot or video still
- one line:
  `Goal: autonomous gate navigation under perception + control constraints`

Timing:

`0:00 - 0:35`

## Slide 2: System Design

Title:

`Learning A Stable Racing Policy`

Main message:

The policy is trained in simulation and controls the drone through a stable local waypoint abstraction.

Say:

`The environment is built on gym-pybullet-drones with custom gates, rewards, randomized tracks, and onboard camera rendering. The policy currently outputs a normalized body-frame waypoint delta and yaw delta. A PID controller tracks that local target. That choice made learning stable, but it is also conservative and not the final MAVLink competition interface.`

Show:

```text
Observation -> MLP policy -> waypoint delta + yaw -> PID -> quadrotor
```

Key bullets:

- action: local waypoint delta plus yaw
- strongest observation: `78`D state vector
- observation includes three-gate lookahead, gate normals, and heading alignment
- caveat: waypoint delta is a training abstraction

Timing:

`0:35 - 1:20`

## Slide 3: Training Pipeline

Title:

`BC -> DAgger -> PPO`

Main message:

The staged pipeline avoids the instability of pure RL from scratch.

Say:

`I start with an expert controller and use it to collect demonstrations for behavioral cloning. Then DAgger rolls out the learned policy and asks the expert for labels on the states the policy actually visits. Finally, PPO fine-tunes from that checkpoint with a behavior-cloning regularizer so it can improve without immediately forgetting how to fly.`

Important clarification:

`The expert is not a minimum-snap trajectory planner. It is an online geometric lookahead controller that targets a point just beyond the next gate while blending future gate directions and gate normals.`

Show:

```text
Expert controller -> BC -> DAgger -> PPO fine-tuning
```

Timing:

`1:20 - 2:05`

## Slide 4: Main Result

Title:

`Representation Was The Biggest Win`

Main message:

The clearest performance jump came from improving what the policy sees, not just more training.

Say:

`The biggest improvement came from representation design. After adding three-gate lookahead, gate normals, and relative heading to the gate plane, the best PPO checkpoint reached 99 percent completion and mean return 82.49 on the harder held-out suite, and 100 percent completion with mean return 82.95 on the easier held-out suite.`

Show:

| Model | Hard Held-out Return | Easy Held-out Return |
|---|---:|---:|
| Earlier PPO champion | `71.68` | lower |
| Richer-observation PPO | `82.49` | `82.95` |

Video:

Use the `10-15` second sim clip here.

Timing:

`2:05 - 3:00`

## Slide 5: What Broke And What Comes Next

Title:

`Honest Limits And Next Steps`

Main message:

The project is strong on the current benchmark, but not yet broadly robust or competition-ready.

Say:

`The harder audits changed the story. The policy stayed strong on the original held-out family, but degraded on mirrored right-turn tracks and most longer six-gate tracks. So the project did not solve general drone racing. It solved a meaningful simulator benchmark and then exposed the next real gaps.`

Then say:

`The next work is already started: a multimodal state-plus-vision branch, a bidirectional training branch, and a position-plus-velocity control branch that better matches MAVLink-style setpoints. Early testing showed that velocity feedforward is not a drop-in speed boost; it needs retraining.`

Show:

- `Strong: original held-out suite`
- `Weak: mirrored/right-turn and longer tracks`
- `Vision: implemented, promising, not yet champion`
- `Control: waypoint delta today, MAVLink position/velocity next`

Timing:

`3:00 - 4:20`

Close:

`The main lesson is that staged imitation-to-RL made flight stable, but representation design and honest evaluation mattered most. The project is now moving from a strong simulator policy toward broader generalization and more competition-aligned perception and control.`

Timing:

`4:20 - 4:30`

## Claims To Avoid

- `The drone is competition-ready.`
- `The policy generalizes broadly.`
- `The vision system is solved.`
- `The expert is minimum-snap.`
- `Waypoint delta is the competition command.`

## Better Phrases

- `competition-motivated`
- `simulator-side research stack`
- `directionally aligned with the latest spec`
- `strong on the current benchmark`
- `harder audits exposed the next gap`
- `multimodal branch implemented but still being evaluated`

## Likely Questions

### Why not pure RL?

Pure PPO was unstable early. BC and DAgger gave the policy a stable flight prior before PPO fine-tuning.

### What was the biggest improvement?

The richer observation design: three-gate lookahead, gate normals, and relative heading.

### Is the expert minimum snap?

No. It is a geometric lookahead expert. A minimum-snap or timed trajectory expert is a strong future upgrade.

### Is waypoint delta what MAVLink expects?

No. It is a learning abstraction. A deployment adapter would map policy outputs into MAVLink position, velocity, acceleration, or attitude targets.

### Is the vision model the best model?

Not yet. The multimodal branch is implemented and promising, but the best completed overall checkpoint is still the richer state-based PPO policy.

## Build Checklist

- Make `5` slides.
- Generate or screen-record the state champion video.
- Add one perception debug screenshot if there is room.
- Keep the result table small.
- Keep the limitations slide honest.
- Rehearse once with a `4:30` timer.

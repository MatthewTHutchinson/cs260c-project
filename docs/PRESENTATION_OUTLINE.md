# 4.5 Minute Presentation Outline

This outline is designed for a short class presentation with a clear technical arc:

1. problem
2. method
3. key result
4. honest limitation
5. next step

The safest talk strategy is:

- frame the problem as an autonomous drone-racing project motivated by the AI Grand Prix / DCL competition setting
- lead with the staged BC -> DAgger -> PPO pipeline
- emphasize the richer-observation improvement as the clearest win
- mention the multimodal vision branch as implemented and promising, but still in progress
- be explicit that broader robustness is not solved yet

Important wording note:

- if you say `Anduril competition`, immediately anchor that to the actual technical source you used:
  `docs/260508_Technical_Spec_0002.pdf`
- the most objective phrasing is:
  `This project is motivated by the AI Grand Prix / DCL autonomous drone racing setting and aligned to the latest technical spec available in the repo.`

## One-sentence thesis

I built an autonomous drone racing pipeline in simulation for an AI Grand Prix / DCL-style competition setting, showed that richer state representation plus imitation-to-RL training improved performance substantially, and then used harder audits to uncover where the policy still does not generalize.

## Time budget

- `0:00 - 0:30` hook and problem
- `0:30 - 1:10` environment and action/observation design
- `1:10 - 2:00` training pipeline
- `2:00 - 3:00` main result
- `3:00 - 3:45` robustness + limitations
- `3:45 - 4:20` perception branch + next steps
- `4:20 - 4:30` close

## Slide-by-slide outline

### Slide 1: Title + problem

Suggested title:

`Autonomous Drone Racing for an AI Grand Prix-Style Competition`

What to say:

`This project is motivated by the AI Grand Prix autonomous drone racing setting, where a drone has to navigate a gate course autonomously under simulator interface constraints instead of being teleoperated. In the latest technical spec I used, the stack is centered around forward-camera perception plus telemetry, rather than access to an easy global planner.`

`My project is the simulator-side learning pipeline for that problem. The main challenge is that pure reinforcement learning is unstable early on, so I used a staged approach: behavioral cloning, then DAgger, then PPO fine-tuning.`

What to show:

- one screenshot of the drone racing environment
- one short context line such as:
  `Competition-style setting: forward camera + telemetry -> perception -> planning -> control`
- one sentence with the BC -> DAgger -> PPO pipeline

### Slide 2: Environment and control design

What to say:

`The simulator is built on top of gym-pybullet-drones with custom gate-crossing events and racing rewards. The policy does not output raw motor commands. Instead, it predicts a normalized body-frame waypoint delta plus yaw delta, and a PID controller tracks that target.`

`The strongest state policy uses a richer 78-dimensional observation: body-frame velocity, angular rates, positions of the next three gates, gate normals, and heading alignment to the next gate plane.`

`This matters for the competition context because the official stack is also perception-plus-control oriented: a forward camera stream, MAVLink telemetry, and no simple global-position shortcut.`

`One limitation is that this waypoint-delta action is a training abstraction, not the final MAVLink interface. In the repo it becomes a nearby position target; future work should map this into MAVLink position targets with velocity feedforward, or eventually train a lower-level velocity or attitude policy.`

What to show:

- one simple diagram:
  `state -> policy -> waypoint delta -> PID -> drone`
- one bullet for action
- one bullet for observation
- optional caveat:
  `Current control: stable local waypoint target; future control: MAVLink position/velocity or attitude targets`

### Slide 3: Training pipeline

What to say:

`The pipeline starts with an expert heuristic policy. First, I collect expert trajectories and train BC. Then I run DAgger to aggregate expert labels on policy-induced states. Finally, I fine-tune with PPO, while keeping a behavioral cloning regularizer to reduce catastrophic forgetting.`

`This staged setup worked much better than trying to do PPO from scratch.`

`Importantly, the current expert is not a spline or minimum-snap planner. It is a hand-built geometric lookahead controller that targets a point just beyond the next gate while blending future gate directions and normals.`

What to show:

- a 3-box diagram:
  `Expert -> BC -> DAgger -> PPO`
- maybe one short line:
  `Goal: stable bootstrapping before aggressive optimization`

### Slide 4: Strongest result

What to say:

`The clearest improvement in the whole project came from changing the representation, not just retuning training. After adding three-gate lookahead, gate normals, and relative heading features, the best PPO checkpoint reached 99 percent completion and mean return 82.49 on the harder held-out suite, and 100 percent completion with mean return 82.95 on the easier held-out suite.`

`That outperformed the earlier policy families by a large margin.`

What to show:

- a small table:

| Model | Hard Held-out Return | Easy Held-out Return |
|---|---:|---:|
| Earlier PPO champion | `71.68` | lower |
| Richer-observation PPO | `82.49` | `82.95` |

- if you have a video or GIF, this is the best place to show it
- best candidate:
  a `10-15` second clip of the richer-observation PPO or expert on one clean held-out track

### Slide 5: Honest robustness story

What to say:

`The important limitation is that good results on the original held-out suite did not mean broad robustness. When I added a harder audit with mirrored right-turn tracks and longer six-gate tracks, the state champion became much less reliable.`

`So the project did not solve general racing in a broad sense. It solved a narrower but meaningful family of simulator courses, and the harder audit exposed the next real gap.`

What to show:

- one bullet:
  `Original held-out suite: strong`
- one bullet:
  `Extended audit: weaker on mirrored/right-turn and long-course tracks`
- one sentence in bold:
  `Benchmark design mattered`

### Slide 6: Perception branch and next steps

What to say:

`I also implemented a perception branch. The simulator now renders onboard RGB images, supports visual randomization, and includes a multimodal state-plus-vision policy. I also tested a detector-based bridge baseline, but that performed poorly, which suggests direct multimodal learning is more promising than reconstructing the old state representation from a lightweight detector.`

`The first multimodal PPO run is promising, but it is not yet the new champion, and the teacher-warmstarted follow-up is still training.`

What to show:

- one onboard camera image
- one line:
  `Vision bridge: weak`
- one line:
  `Multimodal policy: implemented, promising, still in progress`

### Slide 7: Close

What to say:

`The main takeaway is that staged imitation-to-RL training was effective, but the biggest gains came from better representation design. The current best result is strong in-simulation, but the honest next challenge is broader generalization and stronger perception-driven control.`

Short close:

`So the project progressed from stable flight, to strong held-out performance, to a more realistic understanding of where the policy still breaks.`

## Recommended final slide text

`Takeaways`

- `BC -> DAgger -> PPO was a practical training pipeline`
- `Richer observations produced the biggest performance jump`
- `Harder audits exposed real generalization gaps`
- `A multimodal vision branch is now implemented and still evolving`

## What to avoid saying

Avoid these claims unless you are directly asked and want to qualify them carefully:

- `The drone is competition-ready`
- `The multimodal model is already the best model`
- `The policy generalizes broadly`
- `The vision system is solved`

Better replacements:

- `directionally aligned with competition constraints`
- `promising but still in progress`
- `strong on the current simulator benchmark`
- `the harder audit exposed the next weakness`

## If you only have 5 slides

Merge slides like this:

1. problem + environment
2. pipeline
3. strongest result
4. robustness limitation
5. perception branch + next steps

## Likely questions and short answers

### Why not use pure RL?

`Early PPO was too unstable. BC and DAgger provided a strong flight prior that PPO could refine instead of discovering stable flight from scratch.`

### What was the biggest improvement?

`The richer observation design: three-gate lookahead, gate normals, and relative heading features.`

### Is the expert a spline or minimum-snap planner?

`No. The current expert is a geometric multi-gate lookahead heuristic, not a full trajectory optimizer.`

### Is the vision model finished?

`No. The multimodal branch is implemented and promising, but it is still being trained and evaluated more broadly.`

### Is it competition-ready?

`Not yet. The current best completed policies are still simulator-side research models, and the DCL adapter path is still a scaffold rather than a finished runtime client.`

### Is waypoint delta the competition command?

`No. It is my learning abstraction. The competition spec lists MAVLink setpoint messages, so a deployment version would need an adapter that maps policy outputs into position, velocity, or attitude setpoints.`

## Optional speaker script

`I built an autonomous drone racing pipeline in simulation.`

`The core challenge is that pure reinforcement learning is unstable early on, so I used a staged approach: behavioral cloning, then DAgger, then PPO fine-tuning.`

`The environment is built on top of gym-pybullet-drones, but I added custom gate events, shaping rewards, and a waypoint-based action interface.`

`The strongest policy uses a richer observation: not just velocity and gate position, but also three-gate lookahead, gate normals, and relative heading alignment.`

`That representation change produced the biggest improvement in the whole project. The best richer-observation PPO reached 99 percent completion and mean return 82.49 on the harder held-out suite, and 100 percent completion with mean return 82.95 on the easier one.`

`But I also added a harder audit with mirrored right-turn and longer six-gate tracks, and that exposed a real limitation: the policy was much less robust than the original benchmark suggested.`

`Finally, I implemented a perception branch with onboard RGB rendering, visual randomization, and a multimodal state-plus-vision policy. That branch is promising, but it is still in progress and not yet the overall champion.`

`So the project’s main lesson is that staged imitation-to-RL training works, but representation design and honest benchmarking mattered even more than just running more PPO.`

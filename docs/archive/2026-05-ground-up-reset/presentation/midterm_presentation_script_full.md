# Monday Presentation Script

Target: about `4.5` minutes

Tone: technical, conversational, and honest. The goal is to sound like you built the thing, understand its limits, and know exactly what comes next.

## Slide 1: Autonomous Drone Racing

I built an autonomous drone racing pipeline motivated by the AI Grand Prix and Anduril-style competition setting.

The core problem is that the drone has to fly itself. It needs to perceive a gate course, use telemetry and camera input, and send control commands fast enough to make it through the track.

My project focuses on that learning problem in simulation: imitation learning for stable flight, reinforcement learning for improvement, and harder audits to find where the policy still breaks.

## Slide 2: System Architecture

The simulator is built on `gym-pybullet-drones`, but I added the racing task around it: gates, rewards, track layouts, onboard camera rendering, visual randomization, and evaluation tools.

The strongest completed policy is state-based. It sees drone state plus geometric information about the next gates, including gate normals and lookahead.

One important design choice is the action space. The policy outputs a local body-frame waypoint delta and yaw delta, and a PID controller tracks that nearby target.

That made learning much more stable. But it is also a limitation, because a competition system needs MAVLink-style setpoints, where timing, velocity, acceleration, and yaw matter.

## Slide 3: Training Pipeline

First, I collect expert demonstrations and train behavioral cloning, so PPO is not trying to discover flight from scratch.

Then I use DAgger. The learned policy flies, reaches imperfect states, and the expert labels what it should have done there.

Finally, I fine-tune with PPO. I keep a behavioral-cloning regularizer, because otherwise PPO can improve briefly and then forget how to fly.

One clarification: the expert is not minimum snap. It is a geometric lookahead controller that aims through and beyond the next gate using future gate directions and normals.

## Slide 4: Representation Changed The Result

The biggest result was not just from running more PPO. It came from changing what the policy could observe: three-gate lookahead, gate normals, and relative heading to the gate plane.

The best richer-observation PPO checkpoint reached `99%` completion, `0%` crashes, and mean return `82.49` on the harder held-out suite. On the easier suite, it reached `100%` completion and mean return `82.95`.

That is the strongest completed result so far. The video shows that policy flying a held-out course. It is not competition-ready yet, but it shows stable autonomous gate flight.

## Slide 5: Trajectories Show What Generalizes

On the left is the policy on a held-out diamond course. On the right is a different held-out zigzag course with sharper geometry.

This matters because a policy can look good on one four-gate loop and still be overfit. These plots show whether it is actually passing gates, how smooth the path is, and where it cuts corners.

My interpretation is that it learned real gate-following behavior, not just a memorized path. But the course family is still limited, so I do not want to oversell the generalization.

## Slide 6: The Honest Audit

The policy was strong on the original held-out suite, but that did not mean it had solved general drone racing.

When I added mirrored right-turn tracks, performance dropped. When I added longer six-gate courses, several audits failed completely.

So the right claim is not "this solves autonomous drone racing." The right claim is: staged imitation-to-RL produced a strong policy for one meaningful simulator family, and the audits exposed the next gaps.

Now the next training target is clearer: bidirectional courses, longer horizons, more recovery cases, and less dependence on one track style.

## Slide 7: Perception And Control: Next

I added an onboard RGB camera renderer, more realistic gate visuals, and a multimodal state-plus-image policy path. That branch is implemented, but it is not yet the best completed model.

I also need to improve the control interface. The current waypoint-delta action is stable, but conservative. A competition-aligned version should use position, velocity, acceleration, or attitude setpoints with explicit timing.

Finally, the expert should become more time-aware. A minimum-snap or timed trajectory reference would give better demonstrations, especially for faster flight.

My main takeaway is: imitation learning made the project trainable, richer observations produced the biggest performance gain, and harder evaluation showed what still needs to become robust.

## Optional Closing Line

If I had another week, I would prioritize one thing: retraining with competition-aligned control and a harder bidirectional track distribution, then compare state-only, vision-bridge, and multimodal policies on the same audit suite.

## Short Backup Answers

### Is it competition-ready?

Not yet. It is a simulator-side research stack. The DCL/MAVLink runtime adapter and competition-aligned control policy still need more work.

### Is the expert minimum snap?

No. The current expert is a geometric lookahead controller. Minimum-snap or timed trajectory generation is a strong next upgrade.

### Why does it fly slowly?

The current action is a local waypoint target tracked by PID. That is stable, but conservative. To fly faster, the policy and expert need more explicit timing, velocity feedforward, and training on faster trajectories.

### Is waypoint delta what MAVLink expects?

Not directly. Waypoint delta is this project's learning abstraction. MAVLink can accept setpoint-style commands, but the real question is which fields are enabled: position, velocity, acceleration, yaw, yaw rate, or attitude. That choice determines how timing and aggressiveness are handled.

### Is the vision model the best model?

Not yet. The multimodal branch exists and the simulator can render onboard RGB, but the strongest completed checkpoint is still the richer state-based PPO.

### What is the main result?

The main result is that richer geometric observations changed the policy from fragile PPO into a strong state-based racer on the held-out simulator suite: `99%` completion and `0%` crashes on the harder suite.

### What is the main limitation?

The strongest policy still has limited robustness. Mirrored turns and longer six-gate courses exposed directional bias and horizon limits.

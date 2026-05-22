# Monday Presentation Script: Short Version

Target: about `4.5` minutes with room to breathe.

Use this as the main spoken script. Keep [monday_script_full.md](/Users/matthewhutchinson/dev/cs260c-project/docs/presentation/monday_script_full.md) as the longer backup/Q&A version.

## Slide 1: Autonomous Drone Racing

I built an autonomous drone racing pipeline motivated by the AI Grand Prix and Anduril-style competition setting.

The core problem is autonomy: the drone has to perceive a gate course, use telemetry and camera input, and send control commands fast enough to fly the track.

My project studies that learning problem in simulation. I used imitation learning to get stable flight, reinforcement learning to improve the policy, and harder audits to find where it still breaks.

## Slide 2: System Architecture

The simulator is built on `gym-pybullet-drones`, but I added the racing layer: gates, rewards, track layouts, onboard camera rendering, visual randomization, and evaluation tools.

The strongest completed policy is state-based. It sees drone state plus geometric information about upcoming gates, including gate normals and lookahead.

One key design choice is the action space. The policy outputs a local waypoint delta and yaw delta, then a PID controller tracks that target.

That made training stable, but it is also conservative. A real competition system needs MAVLink-style setpoints where timing, velocity, acceleration, and yaw matter more directly.

## Slide 3: Training Pipeline

The training pipeline is behavioral cloning, DAgger, then PPO.

Behavioral cloning gives the policy a reasonable starting point from expert demonstrations.

DAgger lets the learned policy fly, reach imperfect states, and ask the expert what it should have done there.

Then PPO fine-tunes the policy, with a behavioral-cloning regularizer so it does not forget how to fly.

One important detail: the current expert is not minimum snap. It is a geometric lookahead controller that aims through and slightly beyond the next gate.

## Slide 4: Representation Changed The Result

The biggest improvement did not come from just running more PPO. It came from changing what the policy could observe.

I added three-gate lookahead, gate normals, and relative heading to the gate plane.

With that richer observation space, the best PPO checkpoint reached `99%` completion, `0%` crashes, and mean return `82.49` on the harder held-out suite.

On the easier suite, it reached `100%` completion and mean return `82.95`.

This video shows that richer-observation PPO policy flying a held-out course. It is not competition-ready yet, but it shows stable autonomous gate flight.

## Slide 5: Trajectories Show What Generalizes

The video is useful, but trajectory plots make the behavior easier to inspect.

The left plot is a held-out diamond course. The right plot is a held-out zigzag course with sharper geometry.

This matters because a policy can look good on one four-gate loop and still be overfit. The plots show whether the drone actually passes gates, how smooth the path is, and where it cuts corners.

My read is that the policy learned real gate-following behavior, but the track family is still limited.

## Slide 6: The Honest Audit

The harder audit changed the story.

The policy was strong on the original held-out suite, but it did not solve general drone racing.

Mirrored right-turn tracks exposed directional bias. Longer six-gate tracks exposed horizon and recovery limits.

So the honest claim is: staged imitation-to-RL produced a strong policy for one meaningful simulator family, and the audits exposed the next generalization gaps.

## Slide 7: Perception And Control: Next

The next step is closing the gap to the competition setting.

I added onboard RGB camera rendering, more realistic gate visuals, and a multimodal state-plus-image policy path. That branch exists, but the best completed model is still state-based.

I also need better control. The current waypoint action is stable but slow. A competition-aligned version should use position, velocity, acceleration, or attitude setpoints with explicit timing.

Finally, the expert should become more time-aware. A minimum-snap or timed trajectory reference would create better demonstrations for faster flight.

My main takeaway is: imitation learning made the project trainable, richer observations produced the biggest gain, and harder evaluation showed exactly what still needs work.

## If Asked

### Is this competition-ready?

Not yet. It is a simulator-side research stack. The MAVLink runtime and competition-aligned control policy still need work.

### Why is it slow?

The waypoint action is stable but conservative. Faster flight needs explicit timing, velocity feedforward, and faster expert demonstrations.

### Is vision the best model?

Not yet. Vision is implemented, but the best completed checkpoint is still richer state-based PPO.

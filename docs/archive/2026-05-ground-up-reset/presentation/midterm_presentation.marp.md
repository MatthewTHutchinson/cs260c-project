---
marp: true
title: Autonomous Drone Racing V3
description: CS260C Monday presentation deck, polished V3
theme: default
paginate: true
size: 16:9
style: |
  section {
    background: #f8f6ef;
    color: #1f2d31;
    font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
    padding: 54px 70px;
  }
  h1 {
    color: #1f2d31;
    font-size: 50px;
    letter-spacing: -0.02em;
    margin: 0 0 22px;
  }
  h2 {
    color: #244d63;
    font-size: 33px;
    letter-spacing: -0.01em;
    margin: 0 0 18px;
  }
  p, li {
    font-size: 26px;
    line-height: 1.25;
  }
  strong {
    color: #2e6b48;
  }
  .small {
    font-size: 20px;
    color: #5b676b;
  }
  .caption {
    font-size: 18px;
    color: #5b676b;
    margin-top: 7px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1.04fr 1fr;
    gap: 34px;
    align-items: center;
  }
  .gridTop {
    display: grid;
    grid-template-columns: 1.08fr 0.92fr;
    gap: 36px;
    align-items: start;
  }
  .asset {
    border: 1px solid #d2c9b6;
  }
  .tightList li {
    margin-bottom: 12px;
  }
  .diagramOnly {
    display: flex;
    align-items: center;
    justify-content: center;
  }
---

# Autonomous Drone Racing

## A simulator-side learning pipeline for an AI Grand Prix-style setting

![bg right:45%](../../assets/presentation/sim_screenshot.png)

- Staged training: **BC -> DAgger -> PPO**
- Best completed model: richer state-based PPO
- Current focus: robustness, perception, and competition-aligned control

<!--
Speaker notes:
I built this as an autonomous drone racing project motivated by the AI Grand Prix / Anduril competition setting. The important constraint is that this is not a hand-flown drone: the system needs to perceive a course, plan through gates, and send control commands.

For this short talk, I am going to focus on three things: how I got stable flight, what actually improved the learned policy, and what still does not generalize well enough.
-->

---

<div class="diagramOnly">

![w:1160](../../assets/presentation/system_architecture.svg)

</div>

<!--
Speaker notes:
The system starts in simulation. I use gym-pybullet-drones, then add custom race gates, track randomization, richer observations, onboard camera rendering, and evaluation scripts.

The current best policy is state-based. It does not output raw motor thrusts. It outputs a local body-frame waypoint delta and yaw delta, and a low-level PID controller tracks that target. That abstraction made learning stable, but it is not the final competition command interface.
-->

---

<div class="diagramOnly">

![w:1160](../../assets/presentation/training_pipeline.svg)

</div>

<!--
Speaker notes:
Pure PPO was brittle early on, so I used a staged pipeline.

First, I collect expert demonstrations and train behavioral cloning. Then DAgger lets the learned policy visit its own states and asks the expert what it should have done there. Finally, PPO fine-tunes from that warmer start while keeping a BC regularizer so it does not immediately forget how to fly.

One clarification: the current expert is not minimum snap. It is a geometric lookahead controller that aims through and beyond gates using future gate direction and gate normals.
-->

---

# Representation Changed The Result

<div class="grid">
<div>

![w:610](../../assets/presentation/results_table_v3.svg)

</div>
<div>

<video src="../../assets/presentation/state_champion_demo.mp4" poster="../../assets/presentation/sim_screenshot.png" controls muted loop style="width: 100%; border: 1px solid #d2c9b6;"></video>

<p class="caption">Richer-observation PPO on a held-out track.</p>

</div>
</div>

<!--
Speaker notes:
The biggest gain was not just training longer. It was giving the policy better geometry.

The stronger observation space adds three-gate lookahead, gate normals, and relative heading to the gate plane. With that representation, the best PPO checkpoint reached 99 percent completion, 0 percent crashes, and 82.49 mean return on the harder held-out suite. On the easier held-out suite, it reached 100 percent completion and 82.95 mean return.

This is the main positive result of the project.
-->

---

# Trajectories Show What Generalizes

<div class="grid">
<div>

![w:600](../../assets/presentation/state_champion_trajectory.png)

<p class="caption">Held-out diamond course.</p>

</div>
<div>

![w:600](../../assets/presentation/state_champion_trajectory_zigzag.png)

<p class="caption">Held-out zigzag course with sharper geometry.</p>

</div>
</div>

<!--
Speaker notes:
I also wanted the presentation to show more than one happy-path course. The left plot is the current state champion on the held-out diamond, while the right plot is a different held-out zigzag layout.

This is useful because a drone racing policy can look good on one four-gate loop and still be brittle. Trajectory plots make the behavior inspectable: where it cuts corners, how smooth the line is, and whether it is actually passing gates rather than just scoring well accidentally.
-->

---

# The Honest Audit

<div class="grid">
<div>

![w:610](../../assets/presentation/limitations_audit.svg)

</div>
<div>

## What I learned

- The policy is strong on the track family it was trained around.
- Mirrored right-turn tracks exposed directional bias.
- Longer six-gate courses exposed horizon and recovery limits.
- The next benchmark should be harder before I claim “racing robustness.”

</div>
</div>

<!--
Speaker notes:
The harder audits are what made the project more honest.

The policy looked strong on the original held-out suite, but that did not mean broad racing robustness. Once I added mirrored right-turn tracks and longer six-gate tracks, the state champion became much less reliable.

So I would not claim this solves general drone racing. I would claim it solves a meaningful simulator family and gives me a concrete path toward broader robustness.
-->

---

# Perception And Control: Next

<div class="gridTop">
<div>

![w:660](../../assets/presentation/onboard_camera_pov.png)

<p class="caption">Forward onboard camera rendering with realistic gate visuals and scene clutter.</p>

</div>
<div class="tightList">

## Next build targets

- Finish multimodal **state + RGB** training.
- Visualize what the gate detector sees, not just the simulator camera.
- Replace waypoint-only action with competition-aligned setpoints.
- Improve the expert toward timed trajectory planning / minimum-snap style references.
- Retrain on bidirectional, longer, and more varied courses.

</div>
</div>

<!--
Speaker notes:
I also built the first perception branch. The simulator can render onboard RGB, randomize visuals, and train a state-plus-vision policy. That branch is implemented, but it is not yet the best completed model.

The other big next step is control. The current waypoint target is stable but conservative. A deployment-style version should map into MAVLink-like position, velocity, acceleration, or attitude setpoints. That means the learning target and the expert trajectory need to become more time-aware.

My closing takeaway is: staged imitation-to-RL made flight stable, representation design produced the biggest win, and honest evaluation showed exactly what still needs work.
-->

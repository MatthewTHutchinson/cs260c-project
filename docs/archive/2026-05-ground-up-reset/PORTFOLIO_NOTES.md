# Portfolio Notes

## Short version

Built an autonomous drone racing project that teaches a simulated quadrotor to fly through gates using a hybrid imitation-learning and reinforcement-learning pipeline. The system starts from expert demonstrations, improves robustness with DAgger, and then fine-tunes with PPO in a PyBullet drone simulator.

## Audience-friendly explanation

The project tackles a hard robotics problem: getting a drone to race through a sequence of gates without crashing. Instead of training from scratch with reinforcement learning, I first taught the drone to imitate an expert controller. That gave it a stable flight prior. I then used DAgger to expose the model to its own mistakes, and finally used PPO to try to improve speed and performance without forgetting how to fly.

## Technical highlights

- Custom drone racing environment built on top of `gym-pybullet-drones`
- Structured state representation in the drone body frame
- Upgraded observation design with deeper gate lookahead and orientation-aware features
- Onboard RGB camera rendering, gate visuals, and scene randomization inside the simulator
- Detector-backed perception bridge and a first multimodal state+vision policy path
- Competition-facing source reconciliation against the latest AI Grand Prix technical spec, with explicit confidence labels for confirmed vs. historical assumptions
- Action space designed around waypoint deltas instead of raw motor commands
- Sequential BC -> DAgger -> PPO training pipeline
- PPO stabilization work to keep RL from destroying a good imitation policy
- Visualization tools for replaying policies inside the simulator
- Multi-track training and held-out track evaluation to test generalization rather than only memorizing one course
- Randomized-start and harder-track training to improve robustness on unseen layouts
- Dedicated robustness audit covering start-state stress, noise sweeps, and out-of-distribution track geometry

## Good assets to gather

- screenshot or short GIF of the expert flying
- screenshot or short GIF of the BC policy flying
- one training-curve figure
- one concise diagram of the pipeline
- one note showing how PPO was debugged and stabilized

## Suggested portfolio framing

Focus on:

- why pure RL was brittle
- why imitation learning gave the project traction
- how the action and observation design simplified control
- what you learned from debugging PPO rather than only the final score
- how broader training distributions improved robustness on unseen tracks, even when they slightly reduced specialization on easier tracks

## Stronger proof points

- Best overall PPO checkpoint:
  `logs/ppo_generalization_obs_v1/policy_ppo_best.pt`
- Richer-observation hard held-out suite result:
  `99%` completion, `0%` crash rate, aggregate mean return `82.49`
- Richer-observation easy held-out suite result:
  `100%` completion, `0%` crash rate, aggregate mean return `82.95`
- Good narrative:
  the biggest jump came from improving what the policy observed and how the expert planned, not just from more training runs.
- Honest nuance:
  the same audit that showed strong held-out robustness also exposed sharp OOD switchbacks and vertical recovery layouts as the next major failure cases.
- Active extension:
  I implemented and completed a first multimodal state+vision branch rather than only planning it on paper.
- Current multimodal signal:
  the first multimodal PPO run reached best internal mixed validation return `72.75`,
  but a quick external `10`-episode sanity eval reached `90%` completion and `65.69` mean return,
  so it is promising rather than a clean new overall winner.
- Also useful to mention honestly:
  a detector-only perception bridge was much weaker than the multimodal path, so the project exposed a real perception gap instead of pretending that vision was already solved.
- Honest competition gap:
  the latest official spec now clearly defines a MAVLink / UDP + camera-stream interface,
  and the current repo still approximates that in simulation rather than acting as a finished DCL runtime client.
- Current follow-up:
  I started a targeted robustness branch that retrains the full pipeline on new vertical and switchback-inspired tracks rather than only tuning PPO on the old distribution.
- Outcome of that branch:
  it substantially fixed the sharp switchback OOD failures, but it also made the policy slower and slightly worse on the standard held-out tracks.
  That tradeoff is a useful story point because it shows real distribution-shift engineering rather than only cherry-picking one metric.
- Current follow-up:
  I launched a second robustness pass that isolates the remaining drop/recover failure mode instead of continuing to over-weight switchbacks.
- Outcome of that second pass:
  it fixed the drop/recover OOD case, but it broke the switchback gains and hurt nominal performance.
  That was a useful engineering lesson about the limits of simply reweighting training distributions for a single policy.

## One-sentence portfolio summary

Designed and trained a simulated autonomous drone racing agent using behavior cloning, DAgger, and PPO, then substantially improved it through better geometric observations, stronger expert planning, and a new state+vision perception branch rather than only tuning training distribution.

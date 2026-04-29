# Portfolio Notes

## Short version

Built an autonomous drone racing project that teaches a simulated quadrotor to fly through gates using a hybrid imitation-learning and reinforcement-learning pipeline. The system starts from expert demonstrations, improves robustness with DAgger, and then fine-tunes with PPO in a PyBullet drone simulator.

## Audience-friendly explanation

The project tackles a hard robotics problem: getting a drone to race through a sequence of gates without crashing. Instead of training from scratch with reinforcement learning, I first taught the drone to imitate an expert controller. That gave it a stable flight prior. I then used DAgger to expose the model to its own mistakes, and finally used PPO to try to improve speed and performance without forgetting how to fly.

## Technical highlights

- Custom drone racing environment built on top of `gym-pybullet-drones`
- Structured state representation in the drone body frame
- Action space designed around waypoint deltas instead of raw motor commands
- Sequential BC -> DAgger -> PPO training pipeline
- PPO stabilization work to keep RL from destroying a good imitation policy
- Visualization tools for replaying policies inside the simulator
- Multi-track training and held-out track evaluation to test generalization rather than only memorizing one course
- Randomized-start and harder-track training to improve robustness on unseen layouts

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

- Best harder-suite PPO checkpoint:
  `logs/ppo_generalization_v1/policy_ppo_best.pt`
- Harder held-out suite result:
  `100%` completion, `0%` crash rate, aggregate mean return `71.68`
- Same checkpoint on the earlier held-out suite:
  `100%` completion, `0%` crash rate, aggregate mean return `69.57`
- Good narrative:
  the final system did not just memorize one rectangular course; it held up on harder unseen layouts and randomized starts.

## One-sentence portfolio summary

Designed and trained a simulated autonomous drone racing agent using behavior cloning, DAgger, and PPO, with custom control abstractions, RL stabilization tooling, and held-out multitrack evaluation to improve robustness on unseen courses.

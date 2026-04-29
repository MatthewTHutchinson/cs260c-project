# Deep Learning Project Proposal: Hybrid Imitation + RL for Drone Gate Racing (Waypoint-Delta)

## Title
**Bootstrapping Drone Gate-Racing Policies with a Model-Based Expert + DAgger, then PPO Fine-Tuning (Waypoint-Delta Actions)**

## Motivation
Autonomous drone gate racing is a high-speed sequential decision problem where learning directly from low-level motor commands is typically unstable and sample-inefficient. A practical solution is a **hybrid autonomy stack**: use a **classical stabilizing controller** for high-rate tracking, and learn a **mid-level planner** that outputs smooth local navigation targets. This project uses a model-based “expert” to warm-start learning (reducing random exploration) and then improves robustness and speed with **DAgger** and **PPO fine-tuning**.

## Problem Statement
Train a deep policy to navigate a quadrotor through a sequence of gates **in order** and **in the correct direction**, minimizing time while maintaining stability.

**Constraints**
- Gates must be passed **in order** and **correct direction**
- Episode terminates on crash/out-of-bounds/wrong-way traversal/timeout
- Evaluation is on **held-out** randomized tracks/layout seeds

## Core Approach (Single Policy, Sequential Phases)
This project trains **one policy** through phases (no separate “multiple models” required):

1) **Expert (Teacher)**
- Generate feasible trajectories through gates (e.g., spline/minimum-snap through gate centers)
- Track with a stabilizing controller (PID/geometric controller)
- Produce action labels for imitation learning

2) **BC → DAgger (Robustness & Recovery)**
- Behavior Cloning (BC) from expert dataset (fast warm start)
- DAgger: roll out the current policy; query expert on visited states; aggregate dataset; continue training  
  (fixes compounding error and improves recovery)

3) **PPO Fine-Tune (Speed)**
- Initialize PPO from the DAgger policy checkpoint
- Optimize for faster completion while keeping smooth, stable behavior

## Action Space (Locked): Waypoint Delta
Policy outputs a **local waypoint delta** at low frequency (e.g., 10–20 Hz):

**aₜ = [Δx, Δy, Δz, Δyaw]**

Interpretation:
- Δ position is expressed in the **body frame** (recommended) and mapped to world via current attitude
- Clamp deltas to keep targets local and trackable
- A classical tracking controller converts target pose to simulator control inputs

Why waypoint delta:
- Learnable and stable compared to low-level motor commands
- Easy to generate expert labels
- Supports smooth racing behavior with proper reward and curriculum

## Observation Space (State-Based; Vision Separate)
Keep vision separate for the main project to reduce scope. Use sim truth for gate pose.

**Recommended observation vector**
- Drone state subset: velocity (body frame), angular rates, yaw (or heading)
- Relative next gate position in body frame: (dx, dy, dz)
- Lookahead: relative positions to the next 1–2 gates (strong for “racing line” behavior)
- Optional: distance-to-gate, gate normal alignment features

## Environment & Gate Events
- Simulator: PyBullet-based drone racing environment (gate sequence)
- Gates represented as oriented rectangles/planes with a required pass direction
- Gate pass event triggers when:
  - drone crosses the gate plane in the correct direction, and
  - crossing occurs within gate bounds
- Environment emits `gate_passed`, `gate_id`, `next_gate_index` and terminates on invalid events

## Reward Design (Shaped but Safe)
Use dense shaping + strict termination.

**Dense**
- Progress toward next gate (e.g., distance reduction to next gate center / track progress)
- Smoothness penalty: `-λ ||aₜ − aₜ₋₁||`
- Optional stability penalty: large tilt/ang-rate or controller saturation proxy

**Event-based**
- +R for correct gate pass (in order, correct direction)
- Large negative + terminate on crash/out-of-bounds/wrong-way

Anti-hack principles:
- Gate order correctness must dominate progress shaping
- Local deltas bounded
- Terminate on invalid dynamics/violations

## Curriculum Training (How It Fits Without “Multiple Policies”)
Curriculum is not “train many policies.” It’s a **single policy trained on staged difficulty**. You keep the same network weights and continue training as difficulty ramps.

### Curriculum goals
- Make early learning signals dense and stable
- Prevent spending days in “random flailing”
- Gradually introduce the conditions needed for racing speed and robustness

### Recommended curriculum stages
Stage 0 — **Hover / stabilize**
- Fixed target altitude, no gates  
- Success: stable for N seconds without crashes

Stage 1 — **Waypoint tracking**
- No gates; random waypoint targets  
- Success: reach waypoint within tolerance consistently

Stage 2 — **Single gate (straight)**
- One gate, simple approach  
- Success: pass gate with high rate

Stage 3 — **Two gates (mild turn)**
- Add a turn; teach anticipation  
- Success: pass both in order

Stage 4 — **Short course (3–5 gates)**
- Fixed layout, moderate speed  
- Success: complete course reliably

Stage 5 — **Randomized layouts**
- Random gate spacing/angles within bounds  
- Success: generalize across seeds

Stage 6 — **Speed pressure + tighter gates**
- Time penalty increased, gate bounds reduced  
- Success: faster completion without instability

Stage 7 — **Noise/latency/randomization (optional)**
- Dynamics randomization, observation noise, actuation delay  
- Success: robustness under mismatch

### How curriculum is implemented
- **Same policy** continues training
- Change environment parameters based on:
  - a schedule (every N steps), or
  - a performance threshold (e.g., >80% success → advance)
- Store all curriculum parameters in config for reproducibility

### Where curriculum sits in the pipeline
- Use curriculum during **expert data generation** (generate demos for early stages)
- Use curriculum during **DAgger** (the policy learns recovery under increasing difficulty)
- Use curriculum during **PPO fine-tuning** (speed optimization after competence)

## Training Plan (Single-Policy)
### Phase A — Expert dataset
- Generate rollouts across curriculum stages 1–4
- Save `(obs, expert_action)` pairs
- Include perturbed initial conditions for recovery examples

### Phase B — BC then DAgger
- Train BC on expert dataset (warm start)
- Run DAgger iterations:
  - roll out policy in current curriculum stage
  - query expert for labels on visited states
  - aggregate dataset
  - continue training

### Phase C — PPO Fine-tune
- Initialize PPO from the DAgger checkpoint
- Train with shaped reward + curriculum stages 4–6
- Objective: reduce time/steps while maintaining completion rate

## Evaluation Metrics (Report-Ready)
Primary:
- Completion rate (% episodes finish all gates in order)
- Time-to-finish / steps-to-finish
- Crash/out-of-bounds rate

Secondary:
- Smoothness: mean/max `||Δa||`
- Max speed; controller saturation proxy
- Gate approach angle stats (optional)

**No multi-policy ablations required:** evaluate the same policy at checkpoints:
- Expert baseline
- BC checkpoint
- DAgger checkpoint
- PPO fine-tuned checkpoint

## Timeline (5 weeks)
Week 1: Environment + gate events + expert + dataset format  
Week 2: BC warm start + evaluation harness  
Week 3: DAgger loop + recovery improvements  
Week 4: PPO fine-tune + speed + curriculum ramp  
Week 5: Held-out generalization + plots + writeup

## Risks & Mitigations
- Expert too conservative → PPO fine-tune must exist; increase time pressure later
- Reward hacking → strict terminations + bounded deltas + gate order enforcement
- Controller saturation → clamp deltas, penalize action changes, log saturation metrics
- Overfitting → randomize tracks + held-out seeds for evaluation

## Deliverables
- Working sim environment with gate events + logging
- Expert trajectory generator + controller tracker
- Single-policy training pipeline: BC → DAgger → PPO
- Results: success rate + time-to-finish curves across curriculum stages
- Final report with method, metrics, and qualitative rollouts

## Open Decision (one remaining)
Define waypoint delta in:
- **Body frame (recommended):** more learnable and invariant
- World frame: easier to debug but less policy-friendly
# Project Details (Final Spec): Hybrid Imitation + RL for Drone Gate Racing  
**Action Space:** Body-frame waypoint delta  
**Training:** Expert → BC/DAgger → PPO fine-tune (single policy, sequential phases)  
**Scope:** 5 weeks, DL-first, engineering-realistic, vision modular (optional extension)

---

## 1) Project Summary
This project trains a deep policy to fly a quadrotor through a sequence of racing gates **in order** and **in the correct direction**, minimizing time while maintaining stability. The core method is a hybrid autonomy pipeline:

1. **Model-based expert** (trajectory generator + tracking controller) produces competent demonstrations.
2. **Imitation learning (BC → DAgger)** bootstraps the policy and teaches recovery behavior under distribution shift.
3. **PPO fine-tuning** improves speed and “racing line” behavior beyond the expert, addressing the conservative expert ceiling.

This is framed as an AI×engineering project: learning handles mid-level planning; classical control handles stabilization.

---

## 2) Task Definition (Environment + Success Conditions)

### Environment
- Simulator: PyBullet-based quadrotor racing environment (or comparable sim).
- Episode: starts with drone spawned near first gate; ends on success or failure.
- Gates: ordered sequence of oriented rectangles/planes in 3D.

### Success
- Pass all gates **in order**, **within gate bounds**, and **in the correct direction**.
- Report time/steps to completion.

### Failure / Termination
- Collision/crash.
- Out-of-bounds.
- Wrong-direction gate traversal.
- Timeout.

### Gate Events (required)
The environment must output gate events in `info` each step:
- `gate_passed: bool`
- `gate_id`, `gate_index`, `next_gate_index`
- `wrong_way: bool` (optional)
- `collision: bool` (optional)

**Gate pass definition (recommended):**
Let a gate have center `p_gate`, unit normal `n_gate` (direction of correct traversal), and rectangular bounds in the gate frame. A pass occurs when:
1) The drone crosses the gate plane from `signed_dist < 0` to `signed_dist >= 0`, where  
   `signed_dist = (p_drone - p_gate) · n_gate`  
2) The crossing point is within gate bounds (y,z in gate local frame).

This makes gate scoring unambiguous and prevents reward hacks.

---

## 3) Action Space (Locked): Body-Frame Waypoint Delta

### Policy output
\[
a_t = [\Delta x_b,\ \Delta y_b,\ \Delta z_b,\ \Delta \psi]
\]
- \(\Delta x_b,\Delta y_b,\Delta z_b\): delta position in **body frame** (forward, left, up).
- \(\Delta \psi\): yaw delta (radians).

### Transform to world-frame target
Let:
- \(p_w\): current position in world frame
- \(R_{wb}\): rotation mapping **body → world**
- \(\Delta p_b = [\Delta x_b,\Delta y_b,\Delta z_b]^T\)

Then:
\[
p^{target}_w = p_w + R_{wb}\Delta p_b
\]
\[
\psi^{target} = \psi + \Delta\psi
\]

### Clipping (stability-critical)
Clip action to keep targets local and trackable:
- \(\|\Delta p_b\| \le r_{max}\) (start 0.25–0.5 m, increase with curriculum)
- \(|\Delta\psi| \le \psi_{max}\) (start 10–20°)

### Controller interface
A classical tracking controller consumes \((p^{target}_w,\psi^{target})\) and outputs the simulator’s control command format.

**Why body-frame waypoint delta?**
- More invariant and learnable than world-frame targets.
- Naturally supports smooth, local corrections and recovery behaviors.
- Simplifies imitation learning labels (expert can produce local deltas along a trajectory).

---

## 4) Observation Space (State-Based; Vision Separate)

### Philosophy
Keep the main project state-based to maximize training stability and reduce scope. Vision can be an optional extension that replaces “truth gate pose” later.

### Recommended observation vector (body-frame aligned)
At time t, include:
- **Drone dynamics (body frame)**:
  - \(v_b\) (3): body-frame velocity
  - \(\omega_b\) (3): body rates
  - optional: \(a_b\) estimate or last action (helps smoothing)
- **Gate geometry (body frame)**:
  - next gate relative position: \(p^{gate1}_b\) (3)
  - lookahead: \(p^{gate2}_b\) (3) and optionally \(p^{gate3}_b\) (3)
  - optional: gate normal / yaw error features

### Computing relative gate position in body frame
Let \(R_{bw} = R_{wb}^T\). Then:
\[
p^{gate}_b = R_{bw}(p^{gate}_w - p_w)
\]

**Lookahead is recommended** because it enables learned racing lines (setting up for gate k+1 while passing gate k).

---

## 5) Expert (Teacher) Design (Recommended “Lazy MPC”)

### Goal
Generate competent rollouts and meaningful recovery labels without implementing a full nonlinear MPC solver.

### Recommended expert
**Trajectory generator:** spline or minimum-snap trajectory through gate centers (optionally offset for better entry angles).  
**Speed schedule:** heuristic or constraint-based (cap accel/turn rate).  
**Tracker:** PID/geometric controller that tracks the trajectory and yields the same action space as the learner (body-frame waypoint deltas).

### Expert output for imitation
Given current state, expert returns:
- \(a^{expert}_t = [\Delta x_b,\Delta y_b,\Delta z_b,\Delta\psi]\)

How to compute expert delta from a world-frame target along the trajectory:
- pick a trajectory target point \(p^{traj}_w(t+\tau)\) (short horizon)
- compute \(\Delta p_w = p^{traj}_w - p_w\)
- convert to body frame: \(\Delta p_b = R_{bw}\Delta p_w\)
- clip and output \([\Delta p_b,\Delta\psi]\)

**Important:** The expert should be “aggressively feasible,” not overly conservative; PPO fine-tuning is what beats it.

---

## 6) Training Method (Single Policy, Sequential Phases)

You train **one policy** continuously through stages; comparisons are via checkpoints.

### Phase A — Expert Data Collection
- Generate rollouts across easy curriculum stages.
- Log dataset:
  - obs_t
  - expert_action_t
  - done_t
  - optional: gate index, collision flag, progress metrics

### Phase B — Behavior Cloning (BC) Warm Start
- Supervised training on expert dataset (regression loss, e.g., MSE).
- Evaluate on held-out track seeds (no training restart).

### Phase C — DAgger (Core Contribution)
Repeat for K iterations:
1) Roll out current policy.
2) Query expert on states visited by the policy (all steps or when “off track”).
3) Aggregate dataset.
4) Continue supervised training.

DAgger addresses compounding error and improves recovery.

### Phase D — PPO Fine-Tuning (Speed)
- Initialize PPO from the DAgger checkpoint.
- Train on shaped reward with curriculum ramp (later stages).
- Objective: reduce time/steps while preserving success.

---

## 7) Reward Design (Shaped, Safe, Anti-Hack)

### Event-based (dominant)
- +R_gate when a gate is passed correctly (order + direction).
- -R_crash and terminate on crash/out-of-bounds.
- -R_wrong and terminate on wrong-way gate traversal.

### Dense shaping (to provide gradient early)
- Progress to next gate:
  - e.g., `d_prev - d_now` where d is distance to next gate center
- Smoothness:
  - penalize `||a_t - a_{t-1}||`
- Optional stability penalties:
  - large tilt / angular rates / controller saturation proxies

### Anti-hack rules
- Gate order correctness must dominate progress shaping.
- Clip waypoint deltas and enforce realistic bounds.
- Strict termination on invalid behavior.

---

## 8) Curriculum Training (Single Policy, Continuous Ramp)

Curriculum is implemented as changing environment parameters during training, not retraining separate models.

### Recommended curriculum stages
0) Hover stabilize (z hold, no gates)  
1) Waypoint tracking (random targets)  
2) Single gate straight  
3) Two gates mild turn  
4) Short course (3–5 gates fixed)  
5) Randomized layouts (spacing/angles)  
6) Tight gates + stronger time pressure  
7) Noise/latency/dynamics randomization (optional)

### Advancement rule
- Schedule-based (every N steps), OR
- Performance-based (e.g., >80% success rate on eval rollouts)

### Curriculum knobs
- gate size ↓
- gate spacing randomness ↑
- turn severity ↑
- time penalty ↑
- observation noise ↑
- actuation delay ↑
- dynamics randomization ↑

---

## 9) Evaluation Plan (No Multi-Policy Ablations)

You evaluate the **same policy** at checkpoints:
- Expert baseline
- After BC
- After DAgger
- After PPO fine-tune

### Primary metrics
- Completion rate (% episodes completing all gates in order)
- Time-to-finish / steps-to-finish
- Crash/out-of-bounds rate

### Secondary metrics
- Smoothness: mean/max `||Δa||`
- Max speed, tracking error proxies
- Gate approach angle distribution (optional)

### Held-out evaluation
Maintain a fixed set of track seeds/layouts for fair comparisons across checkpoints.

---

## 10) Deliverables (What You Turn In)
- Environment with explicit gate events and logging
- Expert generator (trajectory + tracker) producing waypoint-delta labels
- Single-policy training pipeline: BC → DAgger → PPO
- Results:
  - success rate and time-to-finish across curriculum stages
  - checkpoint comparisons (BC vs DAgger vs PPO) without retraining separate models
- Final report: method, setup, metrics, and qualitative rollouts/plots
- Optional: vision module extension (plug-in replacement for truth gate pose)

---

## 11) 5-Week Timeline (Practical)
**Week 1:** gate events + expert + dataset schema + BC baseline  
**Week 2:** BC training stable + evaluation harness + logging/plots  
**Week 3:** DAgger iterations + recovery improvements  
**Week 4:** PPO fine-tune + curriculum ramp for speed  
**Week 5:** generalization evaluation + writeup + optional extensions

---

## 12) Implementation Recommendations (High Probability of Success)
- Keep policy frequency low (10–20 Hz); controller runs high-rate.
- Use body-frame relative gate geometry + lookahead gates.
- Start with conservative action bounds, widen with curriculum.
- Keep vision out of the main loop unless you have extra time.
- Log everything (run configs, seeds, curriculum stage, reward components).

---

## 13) Remaining Open Choice (Only if you want)
- Choose whether gate lookahead is 2 or 3 gates (recommend 2 for simplicity).
- Choose whether DAgger queries expert on every step or only on “off track” states (start with every step).
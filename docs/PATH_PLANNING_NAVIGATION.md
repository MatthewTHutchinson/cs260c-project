# Path Planning and Navigation

Date: 2026-05-22

## Goal

Build a navigation layer that passes gates in the correct order using FPV detections and telemetry. The first VQ1 goal is reliable completion on a short course with fewer than 10 gates.

The planner should not require GPS, absolute position, depth, or simulator gate labels.

## Navigation Stack

Recommended stack:

```text
gate detector
  -> temporal gate tracker
  -> navigation state machine
  -> local guidance target
  -> control command
```

The planner owns sequence behavior. The controller owns vehicle stabilization.

## VQ1 Strategy

VQ1 is completion-focused. Use a conservative finite-state approach first:

1. `start`: stabilize and find first gate.
2. `approach_gate`: center the gate and move forward.
3. `commit_pass`: continue through the gate when it is large/centered enough.
4. `reacquire_next`: search for the next gate after crossing.
5. `finish`: stop or continue safely after finish detection.
6. `recover`: slow down and search if confidence drops.

This is intentionally not a racing-line optimizer. It is a robust completion machine.

## Gate Order

Rules:

- gates must be passed in correct order
- VQ1 has fewer than 10 gates
- VQ2 has fewer than 20 gates
- same course is used for all teams within a qualifier
- unlimited attempts are allowed during the qualification window

Implication:

We can use repeated attempts to improve our internal understanding of the course, as long as each run remains autonomous and no manual intervention occurs during a run.

## Mapping Policy

There are three levels of map sophistication:

### Level 0: Pure Reactive

Use only the currently visible gate.

Pros:

- easiest to implement
- strong fit for clear VQ1 gates
- no global position required

Cons:

- can fail after gate pass if next gate is temporarily out of view
- weak for sharp turns and occlusions

### Level 1: Local Memory

Maintain short-term relative memory:

- last gate bearing
- estimated pass-through direction
- recent yaw integration
- expected search direction after passing a gate

Pros:

- helps after gate pass
- still does not require a global map

Cons:

- drift accumulates
- not enough for high-speed VQ2 optimization

### Level 2: Course Map Across Attempts

Use logged attempts to build an approximate course graph.

Pros:

- useful for VQ2 speed
- can support racing-line planning

Cons:

- needs careful fair-play review
- no absolute GPS, so map is relative and drift-prone
- may need visual landmarks or simulator-provided local frame behavior

For VQ1, Level 0 plus Level 1 is the right starting point.

## Local Guidance

The planner should convert detection output into a local guidance command:

```text
forward speed
lateral speed
vertical speed
yaw rate
mode
confidence
```

Simple guidance law:

- yaw toward horizontal image error
- move laterally only if yaw alone is insufficient
- climb/descend from vertical image error
- increase forward speed as gate confidence and centering improve
- slow down when gate is near frame edge or detection confidence drops
- enter `commit_pass` when the gate is centered and apparent size exceeds threshold

## Passing a Gate

Without simulator gate events, gate pass detection must come from perception and telemetry.

Possible signals:

- gate apparent size grows then disappears past frame edges
- optical flow / image expansion indicates crossing
- elapsed time after entering commit mode
- body-frame forward motion integrated from velocity telemetry
- next gate appears after old gate loss

Use a conservative combination:

```text
if commit_pass and elapsed_commit_time > threshold:
    advance expected gate index
    enter reacquire_next
```

Tune this with real VQ1 logs.

The current implementation uses a visual proxy before adding a full navigation
state machine:

```text
if near commit gate disappears:
    advance visual sequence index
    search/reacquire

if near commit gate has a far candidate and a close edge candidate:
    advance visual sequence index
    track the far candidate
    ignore close stale candidates briefly
```

The sequence index is a tracker/debug state, not simulator truth. It is derived
from FPV detections and range estimates only.

## Search Behavior

Search should be deliberate, not random.

When detection is lost:

- reduce forward speed
- hold altitude unless telemetry suggests unsafe state
- yaw scan toward the last expected turn direction
- widen search if no detection returns
- avoid rapid oscillation

If repeated attempts reveal a consistent turn direction after a gate, encode that as learned course memory for VQ1.

## Simulator Reset

The active harness should support finite courses:

- explicit start gate
- ordered intermediate gates
- explicit finish gate
- optional loop mode only for legacy baselines
- square gate scoring
- collision with gate frame and boundaries
- no ground-truth labels in perception-facing evaluation

The old cyclic track library can be archived or retained only as a regression baseline.

## VQ2 Direction

VQ2 changes the objective from completion to fastest valid time.

After VQ1 completion is reliable, add:

- course mapping across attempts
- racing-line generation
- time allocation
- velocity-aware trajectory tracking
- obstacle-aware planning
- more aggressive control adapter

Do not start here. VQ1 success is a reliable gate-order navigation problem first.

## Controller Redesign Note

The proportional visual-servo controller is not robust enough for circular and
S-shaped courses. A quick bounded-gain experiment made even the straight
`easy` course worse, which is a useful result: the next improvement should not
be another hand-shaped image-error curve.

The better next method is:

```text
GateTracker sequence state
  -> local gate-passage state machine
  -> short-horizon body-frame waypoint or corridor target
  -> attitude-rate/thrust tracking
```

That gives navigation a notion of "turn toward the next corridor" instead of
asking the low-level controller to infer course geometry from the current gate
center alone.

## Immediate Tasks

1. Implement the navigation state machine.
2. Define the detector-to-navigation message.
3. Implement reactive guidance law.
4. Implement commit/pass/reacquire logic.
5. Build playback tests from logged VQ1 frames as soon as available.
6. Validate finite racecourse semantics in Elodin and later VQ1.
7. Add evaluation metrics for gate order, finish, search time, and detection loss.
8. Replace the visual proxy with an explicit navigation state machine once VQ1
   frames confirm gate appearance and candidate ambiguity.

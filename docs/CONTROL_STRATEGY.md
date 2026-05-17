# Control Strategy Notes

## Current repo action

The current policy action is:

```text
[dx_body, dy_body, dz_body, dyaw] in [-1, 1]
```

`GateRaceAviary` scales this into a local body-frame waypoint:

```text
delta_pos_body = action_xyz * clip_radius
delta_yaw = action_yaw * max_dyaw
target_pos_world = current_pos + body_to_world(delta_pos_body)
target_yaw = current_yaw + delta_yaw
```

That target is passed into the PyBullet `DSLPIDControl` position controller.

Important:

- this is not a confirmed competition command
- this is not directly converted as `velocity = delta / dt`
- it is a short-horizon position target tracked by a PID controller
- the PID internally uses position error, velocity error, integral error, and gravity compensation

## Why this can look slow

The current design was chosen for stability and learnability. It is conservative because:

- `clip_radius` keeps the commanded local target nearby
- the PID damps motion through velocity error
- the expert is a geometric lookahead heuristic, not a time-optimal trajectory planner
- the reward includes smoothness and survival incentives
- PPO is regularized toward the imitation dataset

This is a good way to bootstrap flight, but it likely caps racing speed.

## MAVLink mapping

The latest competition spec confirms support for MAVLink messages including:

- `SET_POSITION_TARGET_LOCAL_NED`
- `SET_ATTITUDE_TARGET`

The closest deployment mapping for the current action is:

```text
SET_POSITION_TARGET_LOCAL_NED:
  position = current_position + body_to_world(action_xyz * clip_radius)
  yaw = current_yaw + action_yaw * max_dyaw
```

A more speed-oriented mapping is:

```text
position = current_position + delta_world
velocity = delta_world / horizon_T
yaw = current_yaw + action_yaw * max_dyaw
```

Here `horizon_T` is a design parameter, not automatically the command timestep.
For example, with `clip_radius = 0.5 m`:

- `horizon_T = 0.05 s` implies up to `10 m/s`
- `horizon_T = 0.5 s` implies up to `1 m/s`

So the horizon controls how aggressively the same waypoint offset is interpreted.

## New repo branch

`GateRaceAviary` now supports:

```yaml
control_mode: position_velocity
velocity_horizon: 0.70
max_target_speed: 2.0
velocity_feedforward_gain: 0.25
```

This keeps the same policy output shape but passes a capped velocity feedforward target into the PID.

The existing state champion was not stable when evaluated directly under this new mode.
That means this should be treated as a retraining branch, not a drop-in speed boost.

## Expert direction

The current expert is not a minimum-snap planner. It is an online geometric lookahead controller.

The next expert upgrade should generate timed trajectories:

```text
gate sequence -> racing line waypoints -> time allocation -> smooth trajectory -> local targets
```

Minimum snap or minimum jerk would be useful here, but only if paired with realistic timing, velocity limits, acceleration limits, and gate-crossing constraints.

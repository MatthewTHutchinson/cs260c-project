# Legacy PyBullet Stack

This folder contains the old CS 260C PyBullet / BC / DAgger / PPO project.

It is preserved for reference, citations, old assets, and possible fallback
experiments, but it is no longer the active project direction.

Use this only when intentionally inspecting the historical simulator or model
training work. The current project lives at the repo root:

```text
algorithm/
docs/
scripts/
```

The reason for this quarantine is simple: the old PyBullet work mixed track
design, gate geometry, rendering, policy training, and evaluation assumptions
too tightly. That made the results hard to trust for the VQ1/final-project
story.

The active direction is now Elodin/VQ1-facing:

```text
FPV gate recognition
  -> temporal tracking
  -> reactive navigation
  -> body-rate/thrust or RC command adapter
```

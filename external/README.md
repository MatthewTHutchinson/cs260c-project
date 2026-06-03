# External Research Checkouts

This folder is for local, unvendored research repositories that support the
current autonomous drone racing project.

The active project should import code from `algorithm/`, not directly from
these checkouts. If a research detector is useful, export or wrap it behind the
existing detector boundary:

```text
FPV frame -> GateObservation candidates -> GateTracker
```

Current ignored local checkouts:

- `external/gatenet/`: upstream GateNet reference clone.

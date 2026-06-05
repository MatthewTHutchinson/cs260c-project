# Project Structure

Date: 2026-06-05

The repo is organized around the current VQ1/local-simulation-facing project.
The old PyBullet training stack has been removed from the active repo.

## Active Project

Use these folders for current work:

```text
algorithm/
assets/presentation/
docs/
learning/
scripts/
external/
```

`algorithm/` contains the simulator-independent racing code:

- gate detection
- temporal tracking
- reactive control
- command-adapter helpers
- frame-convention helpers

`docs/` contains active final-project and VQ1 planning docs.

`learning/` contains the feature-based GRU/MLP behavioral-cloning scaffold for
the T4 training path. It should consume classical CV/tracker features and
allowed telemetry, not raw simulator truth.

`assets/presentation/` contains current final-presentation figures and overlays.

`scripts/` contains current operational scripts for the local-simulation/VQ1 path.

`external/` contains ignored local research checkouts. These are reference
repos, not active runtime packages.

## Reference

Use this for external facts and captured source material:

```text
docs/reference/
```

This includes the current technical spec, MAVLink schema, papers, and captured simulator/VQ1 reference notes.

`docs/reference/gatenet_external.md` records the local GateNet clone, upstream
commit, and practical integration caveats.

## External Harness

The local practice harness is a sibling repo, not vendored into this project:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix
```

This keeps third-party simulator code, Betaflight builds, run databases, and exported telemetry out of the course repo.

## Rule Of Thumb

If the work is part of the final autonomous racing algorithm, put it in `algorithm/`.

If the work is a feature-based learned policy, put it in `learning/`.

If the work explains or supports the final project, put it in `docs/`.

If the work is a current final-presentation visual, put it in `assets/presentation/`.

If the work runs the local simulation/VQ1 workflow, put it in `scripts/`.

If the work is an upstream research checkout, keep it under `external/` and
document the integration path in `docs/reference/`.

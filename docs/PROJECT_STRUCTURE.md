# Project Structure

Date: 2026-05-31

The repo is organized to keep the new VQ1/Elodin-facing project separate from the old PyBullet training stack.

## Active Project

Use these folders for current work:

```text
algorithm/
assets/presentation/
docs/
scripts/
```

`algorithm/` contains the simulator-independent racing code:

- gate detection
- temporal tracking
- reactive control
- command-adapter helpers
- frame-convention helpers

`docs/` contains active final-project and VQ1 planning docs.

`assets/presentation/` contains current final-presentation figures and overlays.

`scripts/` contains current operational scripts for the Elodin/VQ1 path.

## Reference

Use this for external facts and captured source material:

```text
docs/reference/
```

This includes the current technical spec, MAVLink schema, papers, and captured Elodin/VQ1 reference notes.

## Legacy

The old simulator/training stack lives here:

```text
legacy/pybullet/
```

It includes the old configs, tracks, PyBullet environment, training scripts, learned-policy code, evaluation tools, and presentation assets.

Do not add new VQ1/Elodin work there. Only touch it when intentionally inspecting or citing the historical work.

## External Harness

The Elodin practice harness is a sibling repo, not vendored into this project:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix
```

This keeps third-party simulator code, Betaflight builds, run databases, and exported telemetry out of the course repo.

## Rule Of Thumb

If the work is part of the final autonomous racing algorithm, put it in `algorithm/`.

If the work explains or supports the final project, put it in `docs/`.

If the work is a current final-presentation visual, put it in `assets/presentation/`.

If the work runs the local Elodin/VQ1 workflow, put it in `scripts/`.

If the work is about old PyBullet policies, tracks, or training runs, put it in `legacy/pybullet/`.

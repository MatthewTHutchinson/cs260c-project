---
title: "Elodin - The future of aerospace software"
source: "https://www.elodin.systems/post/elodin-ai-grand-prix-race-sim-harness"
author:
  - "[[Written byDaniel Driscoll]]"
published:
created: 2026-05-29
description:
tags:
  - "clippings"
---
## Elodin / AI Grand Prix race sim harness

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a1629fdf6d3038c0096f11d_Slide%2016_9%20-%2019.png)

### Overview

Today we're open-sourcing a practice rig for [Anduril's AI Grand Prix](https://www.theaigrandprix.com/) (a $500K autonomous drone-race competition) so contestants and anyone else curious can start writing autopilot code against a working stack now, before the official Virtual Qualifier 1 simulator drops: [github.com/elodin-sys/ai-grand-prix](https://github.com/elodin-sys/ai-grand-prix). It's open source, runs on macOS and Linux, and the whole setup is uv sync plus a 5-minute Betaflight build. (WSL should be workable too)

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a162d275112f06043e988ad_drone_race_preview.gif)

\- traditional quadcopter simulation development in MathWorks Simulink

### The Elodin Journey

The path here was longer than we expected. I came into this from the games side. One of my earlier jobs was working on The Sims 4, which (whatever else you think of it) does a serious amount of simulation under the hood: emergent behavior, deterministic-ish replay, content pipelines, an editor good enough to ship to gamers, i.e. non-engineers. That polish and joy of use was missing in aerospace tooling, where most teams I talked to were stitching together MATLAB/Simulink + Gazebo + a homegrown Python harness and praying it held together. We started Elodin a few years ago thinking we could close some of that gap, and it turned out to be a much bigger job than any of us understood at the time. Most of the last couple of years has been us sitting next to customers (drone, satellite, missiles, etc), grounding the simulator in real flight data, finding the missing 20% of features that turn a demo into something a flight-software team relies on, and slowly converging on a stack that gets a useful sim up in hours instead of weeks.

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a162a175972daabbeecf3a0_750aedab.png)

\- traditional quadcopter simulation development in MathWorks Simulink

What we ended up with, roughly, is a Rust ECS + JIT-compiled physics core ([nox](https://github.com/elodin-sys/elodin/tree/main/libs/nox)) with Python bindings, a 3D editor that connects to a time-series telemetry DB (elodin-db) over TCP and live-binds GLB models, plots, and camera feeds to ECS components, and a small process runner (s10) so the same Python entry point can spawn whatever else the sim needs (flight controller, render server, external estimator). The interesting bit is that the physics is plain JAX-style code, @el.map functions over typed components, so things like Monte Carlo runs, GPU acceleration, and bit-for-bit deterministic replays fall out naturally. The engine and editor live at [github.com/elodin-sys/elodin](https://github.com/elodin-sys/elodin) under Apache-2.0.

### The Challenge

When the AI Grand Prix was announced earlier this year it landed almost exactly on top of the workflow we'd been refining: physics at one cadence, a real flight controller at its native PID rate, an FPV camera on top, and a single contestant-authored Python function in the middle. Standing up a race-from-scratch program is itself a big undertaking, and as the expected drop date for the official Virtual Qualifier 1 simulator came and went we figured the friendliest thing to do was publish the representative rig we had been building internally, so other teams can start iterating now rather than wait for the starting gunshot in silence.

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a162a175972daabbeecf3a3_41ba67db.png)

Concretely, the practice rig wires three things together with one post-step callback. Elodin runs the 6-DOF rigid-body physics, motor dynamics, drag, ground constraint, multi-rate IMU/baro/mag sensors, and a GPU-rendered 640×360 forward camera tilted +20° to match the published [VADR-TS-002 spec](https://www.theaigrandprix.com/wp-content/uploads/2026/05/260508_Technical_Spec_0002.pdf). Betaflight runs as its real SITL build (master, with ENABLE\_SIMULATOR\_GYROPID\_SYNC so its PID loop blocks on each FDM packet) and handles RC, mixing, and PID exactly the way it does on a real airframe. An ~80-line bridge serializes FDM/RC/PWM packets over UDP and steps both sides in lockstep at 1 kHz, so when you tune a controller against this rig you're tuning against the same Betaflight that will run on the metal. Contestants only edit solver/, which holds a single autopilot(update: SensorUpdate) -> RCCommand function. It gets called every tick with body-frame IMU, world pose, baro, mag, and an optional fresh RGBA camera frame. No GPS, no depth, no motor RPM, matching the official sim's published telemetry contract. We ship a deliberately unimpressive baseline solver (a small altitude/position PID over a hard-coded gate list) so you have something that takes off and clears gates on day one.

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a162a175972daabbeecf3a9_0d7b2e8c.png)

A few caveats up front. On the wire we use Betaflight's UDP packets rather than MAVLink, so anything you build here will need a thin shim to talk to the real qualifier sim once it ships. We currently expose ENU world state to the solver where the spec says NED, and atmospheric effects are a single drag coefficient (no turbulence, no ground effect, no battery sag). The spec also has a small internal inconsistency around the camera FoV (it states VFoV = 90° but its own intrinsics imply VFoV ≈ 58.72° and HFoV = 90°) and we follow the intrinsics, on the bet that the official sim's renderer will too. The repo's ARCHITECTURE.md has a long "Opportunities for improvement" section that is more or less our own todo list, and PRs against any of it are welcome.

### The Result

To try out, you need uv, git, git lfs, and a C toolchain (build-essential + clang-18 on Ubuntu, Xcode CLT on macOS). After that:

`bash scripts/install_elodin.sh # installs the Elodin CLI + DB   uv sync   git submodule update --init --recursive --depth 1 betaflight   bash scripts/build_betaflight.sh   elodin editor sim/main.py`

You should see SUCCESS: SITL integration working! Drone took off! and a \[RACE\] summary in the terminal. The editor opens to a chase cam and FPV viewport you can scrub on the timeline. The whole schematic is plain [KDL](https://kdl.dev/) embedded in sim/main.py, so changing the layout is straightforward. See the KDL API spec here: [https://docs.elodin.systems/reference/schematic/](https://docs.elodin.systems/reference/schematic/)

![](https://cdn.prod.website-files.com/65badeacc94ad9eb3ec9c27e/6a162a175972daabbeecf39d_84ea8e4a.png)

We'd love to hear from people who have built sims for hard-real-time control, anyone planning to enter the AGP (or already in it), and anyone with opinions on the ergonomics of solver/api.py or on lockstep flight-controller integration in general. We're also interested in where this approach falls apart: where the game-engine intuitions stop helping, where Python-as-the-glue hits a wall, where deterministic replay leaks too much abstraction. Happy to go deep on any layer of the stack in the comments.

Thanks for reading; hopefully you find it a useful primer for the competition.

Good luck, see you out there!
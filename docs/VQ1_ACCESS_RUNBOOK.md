# VQ1 Access and Windows Setup Runbook

Date: 2026-05-29

## Situation

VQ1 access is expected to open on Sunday, 2026-05-31.

The simulator is expected to be a downloadable local package, with Windows support and Linux not working according to the April 10 briefing. The first priority is not training. The first priority is to get any valid Windows environment running the simulator, connect to it, and log camera/telemetry/command behavior.

## Best Option Ranking

### 1. A real Windows gaming/workstation machine

Best if available.

Why:

- lowest setup risk
- easiest GPU/DirectX/OpenGL compatibility
- fewer remote desktop/driver surprises
- easier to plug in logs, screen recording, and debugging tools

What to look for:

- Windows 10/11
- NVIDIA GPU preferred
- admin rights to install simulator, drivers, Python, MAVSDK dependencies, and tooling
- stable internet connection for anti-cheat
- enough disk space for simulator, logs, and frame captures

If you can borrow/rent one locally for the first week of VQ1, this is probably worth more than cloud credits.

### 2. UCLA Library CLICC HP Windows laptop / Virtual Desktop

UCLA Library equipment lending says:

- MacBooks and Chromebooks are generally available for 2-week loan periods.
- HP Windows laptops are listed as specialized long-term lending only, up to one quarter.
- UCLA also points to CLICC Virtual Desktop for Windows OS access on any computer.

Source: `https://www.library.ucla.edu/help/services-resources/equipment-lending`

Action:

- Request/ask about an HP Windows laptop immediately.
- Ask whether you can install third-party simulator software.
- Ask whether the machine has a discrete GPU or only integrated graphics.
- Ask whether anti-cheat/local simulator networking will be blocked.

Likely issue:

- Library virtual desktops are convenient for Windows apps, but probably not ideal for a 3D simulator with GPU graphics and custom networking.

### 3. SEASnet Remote Desktop / Remote VDI

SEASnet provides Windows RemoteApp, Remote Desktop, and Remote VDI access through UCLA VPN and a SEASnet account.

Source: `https://www.seasnet.ucla.edu/setting-up-remoteapps-and-remote-desktop/`

Useful facts:

- UCLA VPN is required.
- A SEASnet account is required.
- Remote Desktop provides a full SEASnet lab desktop.
- Remote VDI provides exclusive virtual desktops.
- SEASnet notes inactivity timeouts and resource-use constraints.

Action:

- Confirm your SEASnet account works today.
- Install/test UCLA VPN today.
- Install Microsoft Remote Desktop / Windows App today.
- Email `help@seas.ucla.edu` asking whether a downloadable Windows 3D simulator with UDP networking and possible GPU/graphics requirements can run on Remote VDI or lab PCs.

Likely issue:

- Remote desktops may not expose the right GPU/graphics path.
- You may not have admin install rights.
- Anti-cheat/networking may not work.
- Session timeouts could interrupt runs.

Use SEASnet as a backup/debug environment, not the primary plan unless they confirm it works.

### 4. Windows cloud GPU workstation

Best fallback if no physical Windows machine is available.

Google Cloud has documentation for NVIDIA RTX Virtual Workstations and remote desktop access. Their docs describe virtual workstations for graphics workloads using APIs like Vulkan, OpenGL, and Direct3D, with RDP/HP Anyware/Horizon connection options.

Source: `https://docs.cloud.google.com/compute/docs/gpus/install-grid-drivers`

Recommended cloud target:

- Windows Server or Windows 11 workstation image if available
- NVIDIA L4/G2 or similar graphics-capable VM preferred
- NVIDIA GRID/RTX virtual workstation drivers
- RDP or HP Anyware for interactive graphics

T4 caveat:

- A T4 is useful for ML training, but a downloadable Windows 3D simulator may need a graphics-workstation setup, not just CUDA compute.
- A T4 VM can be worth trying only if the simulator supports that driver/display path.

Budget caveat:

- $50 is enough for experiments, not careless all-week use.
- Use cloud only after local syntax/setup is ready.
- Stop the VM when idle.
- Prefer short validation sessions over leaving it running.

### 5. Buy/borrow used Windows hardware

If the simulator is finicky, a borrowed Windows gaming laptop may beat days of cloud/VDI pain.

Reasonable target:

- Windows 10/11
- NVIDIA GTX/RTX GPU
- 16 GB RAM or more
- admin rights
- wired or stable Wi-Fi

## What To Do Today

### Access Logistics

1. Ask UCLA Library/CLICC about HP Windows laptop availability.
2. Ask whether installation of third-party simulator software is allowed.
3. Ask whether any available Windows laptops have NVIDIA GPUs.
4. Test SEASnet VPN and Remote Desktop access.
5. Email SEASnet about GPU/3D simulator/admin/networking compatibility.
6. Decide whether you have a physical Windows fallback by Saturday night.
7. Prepare a GCP Windows GPU VM fallback plan, but do not burn credits debugging basics.

### Competition Questions To Ask Organizers

Ask these as soon as the VQ1 instructions arrive:

- exact Windows version requirement
- whether VM/cloud execution is allowed
- whether anti-cheat permits cloud/remote desktop
- GPU requirement or minimum graphics API requirement
- whether admin installation is required
- ports/endpoints for MAVLink and camera stream
- whether multiple simulator instances are allowed during VQ1
- whether attempts can be run on different machines

### Repo Work Before Access Opens

1. Implement or prepare a MAVSDK connection probe.
2. Implement or prepare a heartbeat/telemetry logger.
3. Implement or prepare a `HIGHRES_IMU` field logger.
4. Implement or prepare a camera packet/frame logger.
5. Define a log directory format:

```text
logs/vq1/YYYYMMDD_HHMMSS/
  telemetry.csv
  imu.csv
  attitude.csv
  commands.csv
  frames/
  run_meta.json
```

6. Prepare a command probe with tiny, bounded commands:

```text
yaw_rate only
small thrust trim only
small roll_rate only
small pitch_rate only
body velocity fallback if supported
```

7. Prepare a screen recording plan.

## Sunday First-Hour Plan

1. Download simulator and read setup instructions.
2. Install on the most promising Windows machine.
3. Launch simulator without our client.
4. Confirm graphics, login, anti-cheat, and course load.
5. Start our client in passive mode.
6. Confirm heartbeat.
7. Confirm `ATTITUDE`.
8. Confirm `HIGHRES_IMU`.
9. Confirm linear velocity/status telemetry.
10. Confirm camera packet stream and save frames.
11. Do not send aggressive control commands yet.
12. Save a clean baseline log.

## Sunday Second-Hour Plan

1. Send tiny command probes one axis at a time.
2. Record command response.
3. Verify sign conventions:

```text
roll_rate positive
pitch_rate positive
yaw_rate positive
thrust increase
body frame X/Y/Z if using body velocity fallback
```

4. Write down which command interface works best.
5. Start tuning the reactive gate-centering baseline only after telemetry and commands are understood.

## Decision Rule

If a physical Windows machine works, use it.

If no physical machine is available, try:

1. SEASnet Remote VDI only if install/network/GPU are confirmed.
2. GCP Windows graphics VM if cloud is allowed by competition rules and the simulator renders properly.
3. Library HP Windows laptop if install permissions and GPU are adequate.

Do not assume a generic remote Windows desktop can run the simulator until proven.


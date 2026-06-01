# Presentation Assets

Date: 2026-06-01

Current active assets live in:

```text
assets/presentation/
```

## Algorithm Evidence

- `gate_sequence_detected.jpg`: synthetic FPV frame where the gate is visible and the controller commands yaw/forward correction.
- `gate_sequence_commit.jpg`: close-gate frame showing the `commit` state.
- `gate_sequence_search.jpg`: post-loss frame showing fallback to `search`.
- `gate_sequence_trace.png`: time-series plot of mode, confidence, bearing, body-rate/thrust command, and Betaflight RC fields.

These assets were generated from the current algorithm with:

```bash
scripts/inspect_gate_frames.py --demo --demo-frames 42 --out-dir logs/gate_inspection_sequence_v2 --save-mask
scripts/plot_pilot_trace.py \
  --trace logs/gate_inspection_sequence_v2/trace.csv \
  --out logs/gate_inspection_sequence_v2/pilot_trace.png \
  --title "Synthetic Gate Approach: Detection to Control"
```

Then the selected plot/overlays were promoted into `assets/presentation/`.

## Interpretation

Use these as controlled algorithm demonstrations, not real-world performance claims.

They show:

- the detector producing image-space gate measurements
- the tracker switching from `detected` to `commit`, then `tracked`, then `search`
- the controller reducing yaw/roll/pitch commands as the gate centers
- the adapter mapping internal commands into RC-style fields for Elodin/Betaflight

Replace or supplement these with Elodin/VQ1 frame assets as soon as real FPV captures are available.

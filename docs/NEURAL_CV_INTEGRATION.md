# Neural Gate Perception Integration

Date: 2026-06-03

## Decision

Use a learned gate detector as the next perception upgrade, but do not replace
the entire autonomy stack before the final presentation or VQ1 bring-up.

The right boundary is:

```text
FPV frame
  -> classical or neural detector backend
  -> GateObservation candidates
  -> GateTracker
  -> ReactiveGateController / future planner
  -> RacingCommand
```

This keeps the detector swappable while preserving the validated trace from
image measurement to body-rate/thrust command.

## Implemented Hook

The active hook now has two parts:

- `algorithm/neural_gate_detector.py`: OpenCV-DNN/ONNX inference adapter,
  including a named `GateNetONNXDetector` wrapper.
- `algorithm/detector_factory.py`: runtime selection for `classical`,
  `onnx`, or `gatenet` detectors.

Direct construction:

```python
from algorithm.gate_tracker import GateTracker
from algorithm.neural_gate_detector import (
    GateNetONNXDetector,
    NeuralGateDetectorConfig,
)

detector = GateNetONNXDetector(
    NeuralGateDetectorConfig(
        model_path="models/gatenet.onnx",
        output_format="corners8",
    )
)
tracker = GateTracker(detector=detector)
```

Runtime selection:

```bash
CS260C_GATE_DETECTOR=gatenet \
CS260C_GATE_DETECTOR_MODEL=models/gatenet.onnx \
CS260C_GATE_DETECTOR_OUTPUT=corners8 \
scripts/inspect_gate_frames.py \
  --source path/to/fpv_frames \
  --out-dir logs/gatenet_inspection_sim
```

The same environment variables also affect `AutonomousRacingPilot` when no
custom tracker is injected, which lets the local simulation solver swap
detectors without changing solver code.

Supported output layouts:

- `corners8`: four ordered inner gate corners plus confidence
- `bbox`: YOLO-style center, width, height, and confidence/class scores
- `center_distance`: gate center, distance, and confidence
- `heatmap`: segmentation/confidence map for FCN/U-Net style models

Coordinate convention:

- normalized outputs are interpreted in `[0, 1]` image coordinates
- non-normalized outputs default to detector-input pixels and are scaled back
  to the active camera frame
- set `CS260C_GATE_DETECTOR_PIXEL_SPACE=frame` only if the exported model
  already reports full-frame pixel coordinates

The backend returns the same `GateObservation` object as the HSV detector, so
the tracker/controller do not change.

## Open-Source Options

| Candidate | What It Gives Us | Fit For This Project | Caveat |
|---|---|---|---|
| PencilNet | Code, datasets, and linked trained models for gate center, distance, and orientation | Best near-term candidate if we can download/convert a model | TensorFlow/ROS-oriented, trained on its gate/camera domain |
| MonoRaceGate | 2026 drone racing dataset with precise inner-corner labels | Excellent data source for training/fine-tuning a corner detector | Dataset, not a ready inference model |
| GateNet | Shallow gate perception network for wide-FOV fisheye camera | Adapter-ready through `GateNetONNXDetector` | Requires a compatible ONNX export/weights and may need fisheye-to-pinhole care |
| AlphaPilot-style corner detector | Strong architecture: corner confidence maps plus part affinity fields | Best long-term multi-gate geometry idea | Public challenge repos are old and not a clean pretrained package |
| FCN/U-Net gate segmentation | Direct mask/corner signal that can feed PnP/range | Good if weights or quick training data are available | Needs VQ1/local-simulation labels or compatible public weights |
| Viola-Jones cascade | Lightweight historical detector | Low priority | Too brittle for lighting, pose, partial gates, and multiple gates |

Sources:

- [PencilNet repository](https://github.com/open-airlab/pencilnet)
- [GateNet repository](https://github.com/open-airlab/GateNet)
- [MonoRaceGate repository](https://github.com/tudelft/MonoRaceGate)
- [AlphaPilot challenge organization](https://github.com/alphapilotaichallenge)
- [MonoRace paper](https://arxiv.org/abs/2601.15222)

## Why Not Jump Straight To NN-Only Control?

A neural detector solves gate measurement, not gate sequencing or dynamics. The
hard-track failures are mostly navigation failures:

- the reactive controller cuts inside curves
- the tracker has no explicit next-gate identity
- multiple gates in view are ranked by local visual salience, not sequence
- after a missed gate, search can reacquire the wrong gate

So the next high-value work is:

1. use learned CV to produce better gate candidates
2. add sequence-aware candidate selection
3. add future-gate lookahead for curved and S-shaped tracks
4. only then replace the reactive controller with MPC/RL if needed

## Validation Plan

For any neural detector candidate:

1. Export or wrap the model so it outputs one of the supported layouts.
2. Run it offline on saved local-simulation FPV frames.
3. Save overlays with predicted corners/boxes/masks.
4. Compare detector-only metrics against the HSV baseline:
   - visible-gate recall
   - false positives per frame
   - center jitter
   - range/apparent-size stability
   - runtime per frame
5. Use the same local simulation course suite with only the detector swapped.

Success means the detector improves candidate stability without requiring
world pose, simulator gate IDs, GPS, depth, or pre-known gate coordinates.

GateNet status as of 2026-06-03:

- upstream cloned locally as `external/gatenet/` at commit
  `c279bdf8d4e85e40979bbffca783cd4086df5388`
- details recorded in `docs/reference/gatenet_external.md`
- integrated as a named ONNX runtime backend
- not used in the reported local simulation flight results yet
- no GateNet weights are checked into this repo
- first proof should be detector-only overlays, not a live flight claim

The clone confirmed that GateNet is a good architectural match because its
output includes gate center and distance. It is not yet a drop-in detector
because the public repo does not include pretrained weights or an ONNX export.

## Distance-Aware Sequencing

GateNet-style distance estimates are useful for sequencing, but they do not
identify gate order by themselves. The active tracker uses distance as one
piece of visual evidence:

- default to nearest/range-first candidate selection during ordinary approach
- enter `commit` when the current gate is close
- after a near commit, prefer a farther candidate over a very-close edge
  candidate
- briefly ignore close stale detections after a visual pass event
- maintain an internal `sequence_index` for trace/debugging only

This keeps the autonomy boundary legal: no simulator gate IDs, no global pose,
and no pre-known gate coordinates enter the pilot.

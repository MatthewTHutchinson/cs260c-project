# GateNet External Reference Checkout

Date: 2026-06-03

Local checkout:

```text
external/gatenet/
```

Upstream:

```text
https://github.com/open-airlab/GateNet.git
```

Checked commit:

```text
c279bdf8d4e85e40979bbffca783cd4086df5388
```

The checkout is intentionally ignored by Git. It is a local research reference,
not active vendored project code.

## What Is In The Repo

GateNet provides:

- a TensorFlow/Keras shallow CNN architecture in `GateNet/gatenet.py`
- a custom loss in `GateNet/loss.py`
- a Google Drive link for the AU-DR dataset
- a ROS fisheye back-projection package
- example calibration files for its camera setup

The network output is grid-shaped, with five values per cell:

```text
p, cx, cy, distance, orientation
```

This matches our preferred detector boundary well because it can produce gate
center and distance estimates without changing the downstream tracker,
navigation, or command interface.

## What Is Not In The Repo

The upstream checkout does not include:

- pretrained weights
- an ONNX export
- an inference script
- a ready VQ1 camera profile
- a no-ROS runtime adapter

So `GateNetONNXDetector` remains an adapter path, not a live detector claim.
Before using GateNet in flight, we need either compatible weights or a trained
export that maps into one of the supported `algorithm/neural_gate_detector.py`
layouts.

## Camera Caveat

GateNet was designed around a wide-FOV fisheye camera and ROS-style image
geometry. VQ1 is currently documented as a `640 x 360` pinhole camera with no
lens distortion.

This means we should keep both camera profiles:

- `vq1_pinhole`: competition-facing default
- `gatenet_fisheye`: research experiment for GateNet-style perception

Results under the fisheye profile are useful for learned-CV exploration, but
they should not be presented as VQ1-spec results unless the official simulator
camera is changed to match.

## Integration Path

Near-term path:

1. Use the upstream repo as architecture and dataset reference.
2. Train or obtain weights for a GateNet-style center/distance model.
3. Export to ONNX.
4. Configure:

```bash
CS260C_GATE_DETECTOR=gatenet
CS260C_GATE_DETECTOR_MODEL=models/gatenet.onnx
CS260C_GATE_DETECTOR_OUTPUT=center_distance
```

5. Validate detector-only overlays before live flight.

Distance estimates then feed the active `GateTracker` sequence heuristic:

```text
near current gate in commit range
  -> far candidate appears or stale close edge candidate remains
  -> advance visual sequence counter
  -> ignore very-close stale detections briefly
  -> track the farther next-gate candidate
```

# Gate Recognition

Date: 2026-05-22

## Goal

Build a VQ1-first gate recognition stack that works from FPV images and telemetry only. The first target is reliable completion, not perfect 3D reconstruction.

The recognizer should answer:

- where is the next visible gate in the image?
- how confident are we?
- what is the gate bearing relative to the drone?
- can we estimate range well enough for control?
- is this likely a start/finish/race gate distinction?
- did the detector fail, and if so what fallback did navigation use?

## VQ1 Assumptions

From the April 10 notes:

- VQ1 is short and simplified.
- VQ1 has minimal distractions.
- Gates are clearly visible and visually distinguishable.
- Gates are consistent within the round.
- VQ1 has fewer than 10 gates.
- The course can include elevation changes.

From the May 8 spec:

- camera stream is `640 x 360` at `30 Hz`
- camera/body origin is shared
- camera is tilted `20` degrees upward
- camera is pinhole, no distortion
- stated intrinsics are `[320, 320, 320, 180]`
- gate inner opening is `1.5 m x 1.5 m`
- gate outer dimensions are `2.7 m x 2.7 m x 0.26 m`

## Recognition Strategy

Start with a classical detector and only add learned perception if the real VQ1 frames demand it.

Recommended VQ1 stack:

1. Frame receiver and logger.
2. Color/contrast preprocessing.
3. Gate candidate segmentation.
4. Contour or quadrilateral extraction.
5. Candidate scoring.
6. Bearing and approximate range estimation.
7. Temporal tracking.
8. Navigation-facing output with confidence and fallback flags.

This is intentionally simple. VQ1 is expected to have clear gates, and a transparent detector will be faster to debug than a neural detector during the first week.

## Candidate Detector

Initial detector:

- convert image to HSV and/or normalized RGB
- threshold high-salience gate colors after observing real frames
- remove small blobs
- find contours
- fit rectangles or quadrilaterals
- reject candidates with implausible area, aspect ratio, or edge geometry
- rank candidates by area, centrality, shape quality, and temporal consistency

If gates are not strongly color-coded, add:

- edge detection
- rectangular frame template matching
- Hough-style line grouping
- lightweight learned classifier for candidate patches

## Pose and Range Estimation

Minimum output for VQ1 completion:

```text
bearing_horizontal
bearing_vertical
confidence
apparent_size
```

Useful optional output:

```text
range_estimate
gate_plane_hint
corner_pixels
```

Range can be estimated from apparent gate size:

```text
range ~= known_gate_width * fx / apparent_pixel_width
```

This will be noisy but may be enough for slow gate approach. If corners are stable, use a PnP-style estimate with the known square gate dimensions.

## Temporal Tracking

The detector should not make independent decisions every frame.

Maintain a short track state:

- last detection center
- last apparent size
- velocity in image coordinates
- missed-frame count
- confidence decay
- candidate identity score

Fallback behavior must be explicit:

- `detected`: current frame has a valid gate candidate
- `tracked`: no current detection, using short-term temporal prediction
- `search`: no reliable gate; enter scan/search behavior

Do not silently fall back to simulator track truth.

## Start and Finish Gates

April 10 notes say the start and finish gate may differ slightly.

Plan:

- log gate appearance separately for start, intermediate, and finish if visible
- add class labels only if the difference is real and useful
- keep navigation robust if all gates are treated as generic gates

For VQ1, passing gates in correct order matters more than semantic labeling unless the simulator uses visibly distinct start/finish cues.

## Metrics

Offline metrics after VQ1 frame capture:

- detection recall on visible gates
- false positives per frame
- confidence calibration
- bearing error on manually labeled frames
- range-estimate stability
- missed-frame streak length
- frame processing latency

Runtime metrics:

- detections per second
- percentage of control steps using detected/tracked/search state
- number of search recoveries
- last valid detection age at each gate crossing

## Immediate Tasks

1. Build a VQ1 frame logger.
2. Save raw JPEG frames plus decoded RGB frames.
3. Build a small frame-review/labeling utility.
4. Tune the classical detector on real VQ1 frames.
5. Add detector telemetry to evaluation logs.
6. Tune the detector against Elodin FPV frames without world-pose fallback.
7. Keep a simple visual overlay tool for debugging candidates.

## Red Lines

- Do not use simulator gate labels as fallback in competition-facing evaluation.
- Do not assume depth.
- Do not assume GPS or global pose.
- Do not overtrain a neural detector before we see real VQ1 images.

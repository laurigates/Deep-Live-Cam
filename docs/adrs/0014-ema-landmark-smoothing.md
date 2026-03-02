# ADR 0014: EMA Smoothing for Face Landmarks and Bounding Boxes

## Status

Accepted

## Context

Face detection in live webcam mode produces slightly different keypoint (landmark)
positions and bounding box coordinates on each frame, even when the subject is
perfectly still.  This inter-frame jitter manifests as visible wobble or flickering
in the swapped face output, reducing perceived quality.

Ghost, FaceShifter, and other real-time deepfake systems address this with exponential
moving average (EMA) smoothing applied to the detection output before it is consumed
by the swap pipeline.  EMA is a standard temporal filter widely used in computer
vision and robotics for smoothing noisy sensor readings.

Key constraints:

* **Zero latency cost** — EMA is pure arithmetic on existing NumPy arrays; it must not
  measurably reduce live mode FPS.
* **Live mode only** — batch video processing operates on independent frames and should
  use raw detections to preserve accuracy.
* **Per-identity tracking** — a single smoother state must not bleed between different
  faces or between different appearances of the same face after a scene cut.

## Decision

Add a `LandmarkSmoother` class to `modules/face_analyser.py` that:

1. Accepts a configurable `alpha` parameter (EMA weight for the current frame).
2. Maintains per-face state keyed by embedding cosine similarity.
3. On each call to `smooth(faces)`, blends `face.bbox` and `face.kps` with the
   corresponding previous-frame values via:

       smoothed = alpha * current + (1 - alpha) * prev

4. Resets state for any face whose embedding cosine similarity with the stored
   identity falls below `IDENTITY_THRESHOLD = 0.7` (new face entered the scene).
5. Clears all state when no faces are detected (face lost, empty list).

The smoother is instantiated once per webcam session in `create_webcam_preview`
and injected into `_detection_thread_func` as an explicit parameter.  Smoothing
is applied after each call to `detect_faces_for_webcam`, before the result is
written to the shared `detection_result` dict.

The feature is gated behind `modules.globals.landmark_smoothing` (default: `False`)
and `modules.globals.landmark_smoothing_alpha` (default: `0.7`), exposed via:

* `--landmark-smoothing` / `--landmark-smoothing-alpha` CLI flags
* A **Landmark Smoothing** toggle in the Live Mode settings tab of the GUI

## Consequences

### Positive

* Visible reduction in face jitter with default settings (alpha = 0.7 ≈ 30 %
  history blending per frame).
* No FPS regression — the only overhead is a cosine similarity dot product per
  detected face per frame (O(d) where d = embedding dimensionality ≈ 512).
* Disabled by default — zero impact for existing users who do not opt in.
* Runtime-tunable alpha — the user can adjust smoothing strength live without
  restarting the session.
* State resets automatically on face loss or identity change, preventing ghost
  artefacts during fast movement or scene cuts.

### Negative

* Smoothing introduces a positional lag proportional to `(1 - alpha)`.  With the
  default alpha = 0.7, each smoothed coordinate carries 30 % of the previous
  frame's value.  Over fast head movement this is perceptible but acceptable.
* Users who require frame-accurate detection output (e.g. researchers analysing
  jitter) must leave smoothing disabled.
* The identity threshold (cosine ≥ 0.7) is fixed; adversarial lighting changes
  could occasionally cause spurious resets even for the same person.

## Alternatives Considered

### Kalman Filter

Would provide optimal smoothing under Gaussian noise assumptions but requires
tuning noise covariance matrices and adds significant implementation complexity.
EMA achieves similar perceptual quality with a single scalar parameter.

### Savitzky-Golay / Moving Average over N frames

Requires buffering N frames, introducing N-frame latency and higher memory usage.
EMA has equivalent smoothing effect with single-frame memory.

### Smoothing inside `swap_face` rather than in the detection output

The swap pipeline already runs on separate async threads.  Applying smoothing there
would require additional shared state and locks, and would not address bbox-based
enhancement skip-frame decisions which also read the raw detection result.

## References

* InsightFace `Face` class: dict subclass with `__setattr__` that writes to the
  underlying dict — attribute assignment is safe.
* Ghost deepfake system: EMA on keypoints with alpha ≈ 0.7–0.9.
* Issue #67: feature request with proposed API and acceptance criteria.

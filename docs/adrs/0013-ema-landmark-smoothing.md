# ADR 0013: EMA Smoothing for Face Landmarks and Bounding Boxes

## Status
Accepted

## Context

Face detection in live webcam mode produces slightly different keypoint positions and bounding
box coordinates frame-to-frame, even when the subject is stationary. This causes visible
jitter/wobble in the swapped output because the alignment crop position shifts each frame.

Ghost, FaceShifter, and similar real-time deepfake pipelines apply temporal smoothing as a
standard technique to stabilise detection outputs. The simplest effective approach is an
Exponential Moving Average (EMA).

## Decision

Implement a `LandmarkSmoother` class in `modules/face_analyser.py` that applies per-face EMA
to bounding boxes (`.bbox`) and landmark keypoints (`.kps`) across frames.

### EMA formula

```
smoothed_t = alpha * current_t + (1 - alpha) * smoothed_{t-1}
```

Where `alpha ∈ (0, 1]`:
- `alpha = 1.0` → no smoothing (pass-through)
- `alpha = 0.7` → moderate smoothing (default)
- `alpha = 0.3` → heavy smoothing

### Identity tracking

When multiple people are in frame, each face must be matched to its own smoothing history.
We use **embedding cosine similarity** (`normed_embedding` from InsightFace) with a threshold
of 0.7. If the best match is below the threshold the face is treated as a new identity and
starts a fresh smoothing state. This prevents cross-contamination between different subjects.

### Reset conditions

- No faces detected in a frame → all state cleared
- Session ends (webcam stop) → `LandmarkSmoother.reset()` called in cleanup
- Smoothing toggled off during a session → state cleared on next detection cycle

### Scope: live mode only

Smoothing is applied only in the detection thread (`_detection_thread_func` in
`modules/ui_webcam.py`). Batch video processing uses raw frame-by-frame detections
to preserve accuracy for mapping and cluster analysis.

### Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--landmark-smoothing` | `False` | flag | Enable EMA smoothing |
| `--landmark-smoothing-alpha` | `0.7` | (0, 1] | EMA weight for current frame |
| UI toggle: "Landmark Smoothing" | off | — | Live Mode settings tab |

## Consequences

**Positive:**
- Eliminates most frame-to-frame jitter with zero latency penalty (pure arithmetic)
- Per-identity state prevents cross-contamination between subjects
- Configurable alpha allows users to trade responsiveness for smoothness
- Auto-reset on identity change prevents ghost artifacts when face leaves and re-enters

**Negative:**
- With very low alpha (< 0.3), faces moving quickly show noticeable lag
- Embedding-based identity matching requires InsightFace recognition module loaded
  (already a dependency — no extra cost)
- First frame after a reset has no history, so output is raw detection (not a concern in
  practice since jitter is only visible over multiple frames)

# Feature: EMA Landmark Smoothing

## Overview

Apply exponential moving average (EMA) smoothing to face landmark keypoints and
bounding box coordinates across frames in live webcam mode, eliminating the visible
jitter/wobble caused by per-frame variation in face detection output.

## User Stories

- As a **live presenter**, I want the swapped face to stay visually stable when
  I hold still, so that the output looks natural and professional.
- As a **content creator**, I want to reduce output jitter without sacrificing
  responsiveness during head movement, so that the swap tracks my face smoothly.
- As a **power user**, I want to tune the smoothing strength so that I can balance
  stability against tracking responsiveness for my specific use case.
- As a **video processor**, I want batch video processing to be unaffected by this
  feature so that frame-accurate detection output is preserved.

## Acceptance Criteria

- [x] Visible jitter reduction in live webcam mode when the feature is enabled
- [x] Configurable alpha parameter (0.0 = no smoothing, 1.0 = no history) via
      CLI and UI
- [x] No smoothing artefacts when face moves quickly — state resets when identity
      cosine similarity drops below threshold (scene change / new face)
- [x] Disabled by default in batch/video processing mode (live mode only)
- [x] Unit tests for EMA logic, identity matching, and edge cases
- [x] No FPS regression (EMA is pure NumPy arithmetic)

## Non-Functional Requirements

### Performance

- Overhead per frame: one cosine similarity dot product per detected face
  (O(512) floats = ~2 μs on modern hardware at 30 FPS) — negligible.
- Memory: one 512-float embedding + one bbox (4 floats) + one kps array (10 floats)
  per tracked face — negligible.

### Configurability

| Parameter | CLI flag | Global variable | Default | Range |
|-----------|----------|-----------------|---------|-------|
| Enable/disable | `--landmark-smoothing` | `landmark_smoothing` | `False` | bool |
| Alpha (current weight) | `--landmark-smoothing-alpha` | `landmark_smoothing_alpha` | `0.7` | 0.0–1.0 |

### Compatibility

- Disabled by default — zero impact for existing workflows.
- Applies only within the live webcam pipeline (`_detection_thread_func`).
- Batch video processing code path (`process_video`) is unchanged.

## Out of Scope

- Adaptive alpha (automatically adjusting alpha based on face velocity).
  This is a potential future enhancement.
- Smoothing of face probability scores or detection confidence.
- Smoothing in batch/video mode (intentionally excluded for accuracy).

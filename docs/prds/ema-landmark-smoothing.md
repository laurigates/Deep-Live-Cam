# Feature: EMA Landmark Smoothing

## Overview

Add Exponential Moving Average (EMA) smoothing to face landmark keypoints and bounding box
coordinates in live webcam mode to reduce the visible jitter caused by frame-to-frame
detection variation.

## User Stories

- As a streamer using Deep-Live-Cam, I want the swapped face to look stable even when I'm
  not moving, so my output looks professional and not glitchy.
- As a developer, I want to configure the smoothing strength via CLI flag, so I can tune the
  trade-off between responsiveness and smoothness for my hardware.
- As a power user, I want to toggle smoothing on/off without restarting the application, so
  I can compare the effect in real time.

## Acceptance Criteria

- [x] Visible jitter reduction in live webcam mode with default settings (alpha=0.7)
- [x] Configurable alpha parameter (0.0 unavailable — minimum is 0.01, max is 1.0)
- [x] Automatic state reset when face identity changes or no face detected
- [x] Disabled by default — must be enabled via `--landmark-smoothing` flag or UI toggle
- [x] Applied only in live mode (batch/video processing unaffected)
- [x] UI toggle in the Live Mode settings tab
- [x] Unit tests covering EMA math, identity matching, and edge cases
- [x] No FPS regression (EMA is O(n) per face, negligible vs. inference cost)

## Non-Functional Requirements

- **Performance**: EMA computation adds < 0.1 ms per frame (float32 vector ops)
- **Thread safety**: smoother is created per-session and accessed only from the detection
  thread — no shared mutable state across threads
- **Backward compatibility**: feature disabled by default; existing behaviour unchanged

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--landmark-smoothing` | flag | `False` | Enable EMA smoothing |
| `--landmark-smoothing-alpha` | float | `0.7` | EMA weight for current frame |

### Alpha Guide

| Alpha | Effect |
|-------|--------|
| 0.9 | Very responsive, minimal smoothing |
| 0.7 | Moderate smoothing (default) — reduces most visible jitter |
| 0.5 | Heavy smoothing — noticeable lag on fast movement |
| 0.3 | Very heavy smoothing — mostly history |

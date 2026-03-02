# Feature: Ghost Face Swap Model Integration

## Overview

Add Ghost v1/v2/v3 (256×256 resolution) as selectable alternative face swap models alongside the existing inswapper_128. Ghost produces higher-quality swaps at larger resolution, and combined with the existing frame-blending temporal stabilization, reduces video flickering artifacts.

## User Stories

- As a user processing **video files**, I want to select Ghost v3 for higher-quality swaps with better temporal consistency, so that my output videos have fewer flickering artifacts.
- As a user on a **fast GPU**, I want to trade speed for quality by selecting ghost_256_v3, so that I get better output when real-time speed is not required.
- As a user on a **slower machine**, I want to keep inswapper_128 as the default so I don't experience unexpected slowdowns after upgrading.
- As a **developer**, I want to select the swap model via CLI so I can benchmark different models in automated pipelines.

## Acceptance Criteria

- [ ] `--face-swap-model` CLI argument accepts: `inswapper`, `ghost_256_v1`, `ghost_256_v2`, `ghost_256_v3`
- [ ] Default model is `inswapper` (backward compatible)
- [ ] Selected Ghost model is automatically downloaded on first run
- [ ] Ghost model produces valid swapped output (256×256 aligned face pasted back onto frame)
- [ ] All existing frame processors (enhancer, masking) work correctly after Ghost swap
- [ ] Temporal stabilization (frame blending) applies to Ghost output
- [ ] All execution providers (CPU, CUDA, CoreML) work with Ghost models
- [ ] No regression on existing inswapper behavior
- [ ] Integration tests cover Ghost model loading and swap pipeline

## Non-Functional Requirements

### Performance
- Ghost v1: Expected ≥15 FPS on NVIDIA RTX 3060 equivalent
- Ghost v3: Expected ≥8 FPS on NVIDIA RTX 3060 equivalent
- Model download: < 5 minutes on broadband connection (739 MB max)

### Reliability
- Graceful fallback to inswapper if Ghost download fails
- Clear error message if unsupported provider/Ghost combination detected
- SHA-256 checksum verification for downloaded Ghost models

### Compatibility
- Python 3.13 only (project requirement)
- ONNX Runtime ≥ 1.16.0
- macOS ARM64, Linux x86_64, Windows x86_64

## Model Selection UX

### CLI
```bash
# Default (inswapper)
uv run run.py -s source.jpg -t target.mp4 -o output.mp4

# Ghost v3 (highest quality)
uv run run.py -s source.jpg -t target.mp4 -o output.mp4 --face-swap-model ghost_256_v3

# Ghost v1 (faster)
uv run run.py -s source.jpg -t target.mp4 -o output.mp4 --face-swap-model ghost_256_v1
```

### Justfile
```bash
just start-ghost           # Start with ghost_256_v3 (quality preset)
just start-ghost model=ghost_256_v1  # Start with specific variant
```

## Temporal Stabilization Behavior

The existing `interpolation_weight` / `enable_interpolation` settings control temporal blending:
- `enable_interpolation=True`, `interpolation_weight=0.0` — full current-frame weight (no blending)
- `enable_interpolation=True`, `interpolation_weight=0.3` — 30% previous frame, 70% current

Ghost's "temporal stabilization" in the issue context refers to this post-inference blending, which is already implemented in the codebase and applies equally to Ghost models.

## Out of Scope

- Recurrent/stateful Ghost ONNX models (none exist in FaceFusion's model repo)
- Ghost model fine-tuning or retraining
- HyperSwap integration (separate issue)
- Ghost model for live webcam mode (to be evaluated based on FPS benchmarks)

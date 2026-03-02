# ADR 0012: Ghost Face Swap Model Integration

## Status
**Proposed** (March 2026)

## Context

The current face swapper uses `inswapper_128_fp16.onnx` (InsightFace's INSwapper), which operates at 128×128 resolution. The primary visible artifact in video output is frame-to-frame flickering due to independent per-frame inference with no temporal awareness.

**Ghost** (AI Forever / Sber AI) is an alternative GAN-based face swap architecture using an AEI-Net (Adaptive Embedding Integration) generator with Ghost bottleneck blocks. It operates at 256×256 resolution and has been ONNX-exported by FaceFusion into three variants (v1/v2/v3).

### Alternatives Compared

| Model | Resolution | VRAM | Quality | Speed | Temporal |
|-------|-----------|------|---------|-------|---------|
| inswapper_128_fp16 | 128×128 | ~500 MB | Baseline | Fastest | None |
| ghost_256_v1 | 256×256 | ~700 MB | Better | Moderate | Blending |
| ghost_256_v2 | 256×256 | ~800 MB | Better+ | Moderate | Blending |
| ghost_256_v3 | 256×256 | ~900 MB | Best | Slower | Blending |

*Note: Temporal stabilization is applied post-inference via frame blending, not via recurrent state in the ONNX graph.*

### Key Architectural Differences from inswapper

| Aspect | inswapper_128 | ghost_256_v* |
|--------|--------------|--------------|
| Input resolution | 128×128 | 256×256 |
| Face alignment | FFHQ 512 → 128 | ArcFace 5-point → 256 |
| Source embedding | `normed_embedding @ emap` | `normed_embedding` (direct) |
| Session type | InsightFace model zoo | Direct ONNX InferenceSession |
| Normalization | `/ 255, mean/std` | `/ 127.5 - 1.0` (→ [-1,1]) |
| Output range | [0,1] | [-1,1] (denormalize: `* 0.5 + 0.5`) |

## Decision

Integrate Ghost v1/v2/v3 as selectable swap models alongside inswapper_128, controlled via `--face-swap-model` CLI flag (default: `inswapper`).

**Implementation approach:**

1. **`GhostSwapper` class** in `face_swapper.py` — wraps an `onnxruntime.InferenceSession` and exposes a `.get(img, target_face, source_face, paste_back=False)` interface compatible with the existing `swap_face()` pipeline.

2. **Model selection via globals** — `modules.globals.face_swap_model` (str) selects which model to load. `get_face_swapper()` branches on this value.

3. **Pre-paste upscaling** — Ghost produces 256×256 crops; the existing `_paste_scale_from_M()` / `_upscale_crop_for_paste()` pipeline handles this correctly (k is computed from M regardless of crop size).

4. **Temporal stabilization** — The existing `_get_previous_frame()` / `_set_previous_frame()` blending in `apply_post_processing()` provides temporal smoothing for Ghost just as it does for inswapper. No additional state management is needed.

5. **Model download** — Ghost ONNX models are sourced from `facefusion/facefusion-assets` GitHub releases. `pre_check()` selectively downloads only the selected model variant.

## Consequences

### Positive
✓ **Higher quality** — 256×256 resolution produces sharper, more detailed swaps
✓ **Temporal consistency** — Combined with existing frame blending, reduces flickering
✓ **Backward compatible** — `inswapper` remains the default; no breaking changes
✓ **Plugin-compatible** — GhostSwapper conforms to the same `.get()` interface as INSwapper
✓ **Selectable** — Users choose quality/speed tradeoff via CLI or UI
✓ **All providers** — Ghost ONNX runs on CPU, CUDA, CoreML, ROCm equally

### Negative
✗ **Model size** — ghost_256_v3 is ~739 MB vs ~180 MB for inswapper_128_fp16
✗ **Speed** — 256×256 inference is slower; ~20-40% FPS reduction on same GPU
✗ **No batch inference** — Ghost uses direct ONNX session (no INSwapper batching)
✗ **Download dependency** — FaceFusion model URLs may change; checksums needed
✗ **ONNX tensor names** — Exact input/output names require verification by inspecting downloaded model

### Mitigations
- Ghost is opt-in; inswapper remains default and unaffected
- FPS impact documented in README; users warned about expected slowdown
- Model URLs pinned to a specific release tag with SHA-256 checksum verification
- `GhostSwapper.get_inputs()` introspects tensor names at load time

## Evidence

### Ghost Repository
- Source: `https://github.com/ai-forever/ghost` (AEI-Net generator with AAD blocks)
- Training: 256×256 face crops with ArcFace 512-D identity embedding input

### FaceFusion Integration
- FaceFusion supports Ghost ONNX models (`ghost_256_v1/v2/v3`) as selectable face swapper models
- Models exported to ONNX and hosted at `facefusion/facefusion-assets` releases

### Existing Infrastructure Reused
- `_paste_back()` — works for any crop size
- `_upscale_crop_for_paste()` — scale-adaptive based on M matrix
- `_get_previous_frame()` / `_set_previous_frame()` — temporal blending already present
- `conditional_download()` / `download_model_if_needed()` — model download infrastructure

## Related Decisions
- [ADR 0001: ONNX + InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md)
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md)
- [ADR 0007: Model Fallback and Switching](0007-model-fallback-and-switching-mechanism.md)

**Last Reviewed**: March 2, 2026 | **Decision Maker**: laurigates

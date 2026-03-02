# PRP: Implement Ghost Face Swap Model Integration

## Objective

Integrate Ghost v1/v2/v3 ONNX models as selectable face swap alternatives in Deep-Live-Cam, exposing model selection via `--face-swap-model` CLI flag. Ghost operates at 256×256 and uses direct ArcFace embedding input (no emap transform).

## Background

See: [ADR 0012](../adrs/0012-ghost-face-swap-model-integration.md) | [PRD](../prds/ghost-face-swap-model.md)

**Ghost ONNX model interface:**
- Input 0: target face crop, shape `[1, 3, 256, 256]`, dtype `float32`, range `[-1, 1]`
- Input 1: source ArcFace embedding, shape `[1, 512]`, dtype `float32`, L2-normalized
- Output 0: swapped face, shape `[1, 3, 256, 256]`, dtype `float32`, range `[-1, 1]`

Preprocessing: `blob = (rgb_crop / 127.5) - 1.0`, transpose to NCHW
Postprocessing: `bgr = clip((nchw_out * 0.5 + 0.5) * 255)[:, :, ::-1]`

Face alignment: 5-point arcface alignment to 256×256 (via `insightface.utils.face_align.norm_crop2`)

## Implementation Steps

### Step 1: Add configuration variables

**`modules/globals.py`** — Add after `face_swapper_enabled`:
```python
face_swap_model: str = 'inswapper'  # 'inswapper' | 'ghost_256_v1' | 'ghost_256_v2' | 'ghost_256_v3'
```

**`modules/processing_config.py`** — Add to `ProcessingConfig` dataclass:
```python
face_swap_model: str = 'inswapper'
```

**`modules/processing_config_factory.py`** — Add to both `build_config_from_globals()` and `build_config_from_cli_args()`:
```python
face_swap_model=modules.globals.face_swap_model,
```

### Step 2: Add CLI argument

**`modules/core.py`** — In `parse_args()`, add:
```python
program.add_argument(
    '--face-swap-model',
    help='face swap model to use',
    dest='face_swap_model',
    default='inswapper',
    choices=['inswapper', 'ghost_256_v1', 'ghost_256_v2', 'ghost_256_v3'],
)
```
And in the args assignment block:
```python
modules.globals.face_swap_model = args.face_swap_model
```

### Step 3: Implement GhostSwapper class

**`modules/processors/frame/face_swapper.py`** — Add `GhostSwapper` class and `_GHOST_MODELS` registry.

### Step 4: Update pre_check() for Ghost model download

In `pre_check()`, after the existing inswapper download block, add Ghost download logic.

### Step 5: Update get_face_swapper() to support Ghost

Add a branch before the existing insightface model loading to load the Ghost ONNX model.

### Step 6: Update pre_start() for Ghost

Check the Ghost model path exists when Ghost is selected.

## Success Criteria

- [ ] `pytest tests/test_ghost_swapper.py` passes with 0 failures
- [ ] `uv run run.py --face-swap-model ghost_256_v3 -s s.jpg -t t.mp4 -o o.mp4` runs end-to-end
- [ ] Existing tests pass: `pytest -x -q` green
- [ ] Ghost swap output is a valid 256×256 face pasted back onto frame
- [ ] `inswapper` (default) produces identical results before/after this change

## Testing Strategy

### Unit tests (`tests/test_ghost_swapper.py`)
1. **GhostSwapper instantiation** — mock ONNX session, verify .get() returns (bgr, M) tuple
2. **Preprocessing correctness** — blob range [-1, 1], NCHW shape [1, 3, 256, 256]
3. **Postprocessing correctness** — output range [0, 255], shape [256, 256, 3], BGR
4. **Default model unchanged** — `globals.face_swap_model == 'inswapper'` by default
5. **CLI argument accepted** — `--face-swap-model ghost_256_v3` parses correctly
6. **ProcessingConfig field** — `ProcessingConfig(face_swap_model='ghost_256_v1')` works

### Integration tests
1. **Ghost pre_check** — mock conditional_download, verify correct URL called for each variant
2. **Swap pipeline** — mock GhostSwapper.get(), verify _paste_back is called with 256×256 crop

## Known Risks

1. **FaceFusion model URL stability** — URLs are versioned to a release tag; if FaceFusion changes the tag, downloads will fail. Mitigation: pin to verified tag + checksums.

2. **ONNX tensor names** — Actual tensor names require inspecting the downloaded model. The implementation uses `session.get_inputs()[0].name` introspection to avoid hardcoding.

3. **CoreML compatibility** — Ghost ONNX models may fail on CoreML EP due to unsupported ops. Fallback to CPU if CoreML fails.

4. **Batch inference** — Ghost uses direct ONNX session; the existing `batch_swap_faces()` path cannot be reused. This reduces throughput for `--many-faces` mode. Document as known limitation.

## File Summary

| File | Change Type | Description |
|------|------------|-------------|
| `modules/globals.py` | Modify | Add `face_swap_model = 'inswapper'` |
| `modules/processing_config.py` | Modify | Add `face_swap_model: str = 'inswapper'` |
| `modules/processing_config_factory.py` | Modify | Wire `face_swap_model` in both factories |
| `modules/core.py` | Modify | Add `--face-swap-model` CLI argument |
| `modules/processors/frame/face_swapper.py` | Modify | Add `GhostSwapper` class, update loader |
| `tests/test_ghost_swapper.py` | Create | Unit tests for Ghost integration |
| `docs/adrs/0012-*.md` | Create | Architecture decision record |
| `docs/prds/ghost-face-swap-model.md` | Create | Product requirements |
| `docs/prps/implement-ghost-face-swap-model.md` | Create | This document |

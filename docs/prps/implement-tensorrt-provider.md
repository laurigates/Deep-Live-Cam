# PRP: Implement TensorRT Execution Provider with Engine Caching

## Objective

Add `TensorrtExecutionProvider` support to Deep-Live-Cam so that NVIDIA GPU users can achieve 2–5× faster face swapping compared to the `cuda` provider, with a persistent engine cache in `models/trt_cache/` to avoid repeated compilation overhead.

## Background

ONNX Runtime's TRT EP applies TensorRT graph fusion, kernel auto-tuning, and FP16 precision automatically.  The main user cost is a one-time engine compilation (30–120 s per model) on first use.  ONNX Runtime caches compiled engines to disk and loads them directly on subsequent runs.

The existing `CoreMLExecutionProvider` integration in `modules/onnx_providers.py` provides an established pattern to follow: detect the EP, build a platform-specific options dict, and return a `(name, options)` tuple instead of a plain string.

## Implementation Steps

### 1. Create `modules/tensorrt_cache.py`

New module responsible for:
- `TRT_CACHE_DIR` constant pointing to `models/trt_cache/`
- `get_cache_dir()` — creates and returns the cache directory
- `has_cached_engines()` — detects `.engine` files (used for first-run UX)
- `build_tensorrt_provider_options()` — returns the options dict with FP16 + caching

Key options to set:

| Option | Value | Reason |
|--------|-------|--------|
| `trt_fp16_enable` | `1` | ~2× throughput with identical numerical output |
| `trt_engine_cache_enable` | `1` | Persist engines across sessions |
| `trt_engine_cache_path` | `models/trt_cache/` | Deterministic, project-local cache |
| `trt_max_workspace_size` | `2 * (1 << 30)` | 2 GiB, sufficient for all current models |
| `trt_min_subgraph_size` | `3` | Avoid delegating trivial subgraphs to TRT |

### 2. Update `modules/onnx_providers.py`

Extend `build_providers_config()` to handle `TensorrtExecutionProvider`:

```python
elif p == "TensorrtExecutionProvider":
    from modules.tensorrt_cache import build_tensorrt_provider_options
    config.append(("TensorrtExecutionProvider", build_tensorrt_provider_options()))
```

This is automatically picked up by:
- `face_swapper.py` (already calls `build_providers_config`)
- `_onnx_enhancer.py` (already calls `build_providers_config`)

### 3. Update `modules/processors/frame/face_enhancer.py`

`get_face_enhancer()` currently passes the raw `providers` list directly to `onnxruntime.InferenceSession`.  Update it to call `build_providers_config(providers)` first, matching the pattern in `face_swapper.py`.

Import to add:
```python
from modules.onnx_providers import build_providers_config
```

Change in `get_face_enhancer()`:
```python
_providers = providers if providers is not None else modules.globals.execution_providers
providers_config = build_providers_config(_providers)
FACE_ENHANCER = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers_config)
```

### 4. Update `modules/core.py`

**`suggest_execution_threads()`**: Return `1` when TRT EP is active (TensorRT handles GPU parallelism internally).

**`release_resources()`**: Include `TensorrtExecutionProvider` in the CUDA cache-clear check (both share the same GPU memory pool via torch).

**`pre_check()`**: After existing checks, when TRT EP is selected and no cached engines exist, print a status message warning about first-run compilation time.

### 5. Write Tests — `tests/test_tensorrt_cache.py`

| Test class | What it verifies |
|-----------|-----------------|
| `TestGetCacheDir` | Creates directory when absent; returns path without error when present |
| `TestHasCachedEngines` | False when dir absent; False with no `.engine` files; True when engine file present |
| `TestBuildTensorrtProviderOptions` | Returns dict with `trt_fp16_enable=1`, `trt_engine_cache_enable=1`, string cache path, positive workspace size |
| `TestBuildProvidersConfigTensorRT` | TRT EP becomes `(name, opts)` tuple; CUDA/CPU pass through; fallback chain ordering preserved |
| `TestFaceEnhancerUsesProvidersConfig` | `face_enhancer` imports and calls `build_providers_config` |

## Dependency Requirements

No new Python packages are required for the code changes.  TensorRT support is built into `onnxruntime-gpu` when the host system has TensorRT 8+ installed.

NVIDIA users who want TensorRT must install TensorRT separately:

```bash
# NVIDIA CUDA + TensorRT setup (system-level, not pip):
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/
```

Or use the NVIDIA container toolkit which includes TensorRT out of the box.

## Success Criteria

- [ ] `uv run pytest tests/test_tensorrt_cache.py -v` passes with 0 failures
- [ ] All existing tests continue to pass: `uv run pytest -x -q`
- [ ] On an NVIDIA system: `uv run run.py --execution-provider tensorrt` loads without error
- [ ] Second run is ≥10× faster to start than first run (engine cache hit)
- [ ] FPS with TRT EP ≥ 2× FPS with CUDA EP on `inswapper_128_fp16.onnx`

## Testing Strategy

### Unit tests (CI-safe, no GPU required)
- Mock `get_cache_dir()` with `tmp_path` to avoid touching the real filesystem
- Patch `onnxruntime.InferenceSession` to verify providers are configured correctly
- These run in the default `pytest` suite with no special markers

### Integration tests (GPU required, marked `@pytest.mark.integration`)
- Load the actual ONNX model with TRT EP on an NVIDIA machine
- Verify startup time < 30 s on second run (cache hit)
- Measure FPS over 300 frames with and without TRT EP

### Benchmark tests (marked `@pytest.mark.benchmark`)
- Use `pytest-benchmark` to compare frame throughput across providers
- Run with `pytest -m benchmark` on NVIDIA CI runners only

## Rollout Notes

- TRT EP is opt-in via `--execution-provider tensorrt`; no existing behaviour changes.
- The `models/trt_cache/` directory should be added to `.gitignore` (compiled engines are binary and system-specific).
- Document the provider in `README.md` under the execution provider section.

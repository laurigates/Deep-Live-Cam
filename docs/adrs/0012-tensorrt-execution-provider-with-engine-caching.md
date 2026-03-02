# ADR 0012: TensorRT Execution Provider with Engine Caching

## Status

Accepted

## Context

Deep-Live-Cam already supports multiple ONNX Runtime execution providers (`cpu`, `cuda`, `coreml`, `rocm`).  NVIDIA users running the `cuda` provider benefit from GPU acceleration, but a significant additional speedup is available through ONNX Runtime's `TensorrtExecutionProvider` (TRT EP).

TRT EP applies TensorRT graph fusion, kernel auto-tuning, and FP16 precision selection automatically.  Benchmarks (FaceFusion project, NVIDIA documentation) show 2–5× throughput improvement over CUDA EP for the same ONNX models:

| Model | Expected speedup |
|-------|-----------------|
| `inswapper_128_fp16.onnx` | 2–3× |
| `gfpgan-1024.onnx` | 3–5× |
| InsightFace detection | ~2× |

The main cost is a one-time TensorRT engine compilation (30–120 s per model) the first time a model runs under TRT EP.  Compiled engines can be cached to disk so subsequent runs skip compilation.

### Constraints

- TensorRT is only available on NVIDIA GPUs running Linux or Windows.  macOS (Apple Silicon / Intel) and AMD users are unaffected.
- ONNX Runtime's `onnxruntime-gpu` package already ships TensorRT support when a compatible TensorRT installation is present.  No additional pip package is required for most NVIDIA Linux setups.
- A single ONNX Runtime variant (`onnxruntime-gpu`) must be installed — mixing `onnxruntime` and `onnxruntime-gpu` in the same environment causes import errors (ADR 0004).

## Decision

Add `tensorrt` as a selectable execution provider via `--execution-provider tensorrt` CLI flag.  Implement persistent engine caching in `models/trt_cache/` so that compiled engines survive across sessions.

### Provider configuration

`modules/tensorrt_cache.py` owns all TRT-specific configuration:

- `get_cache_dir()` — returns and creates `models/trt_cache/`
- `has_cached_engines()` — detects whether compiled engines already exist (used for first-run UX warning)
- `build_tensorrt_provider_options()` — returns the options dict for `TensorrtExecutionProvider`:

```python
{
    "device_id": 0,
    "trt_fp16_enable": 1,
    "trt_engine_cache_enable": 1,
    "trt_engine_cache_path": "<models/trt_cache/>",
    "trt_max_workspace_size": 2 * (1 << 30),  # 2 GiB
    "trt_min_subgraph_size": 3,
}
```

`modules/onnx_providers.build_providers_config()` is extended to recognise `TensorrtExecutionProvider` and attach the options dict, following the same pattern already used for `CoreMLExecutionProvider`.

### Cache invalidation

ONNX Runtime's TRT EP generates engine file names using a hash of the model graph and input shapes.  If the model file changes (e.g. the user downloads a newer version), ORT automatically compiles a new engine.  The old engine file remains in the cache directory but is never loaded, so no manual hash tracking is needed.

### Thread count

`core.suggest_execution_threads()` returns `1` when TRT EP is selected.  TensorRT performs GPU-parallel execution internally; extra CPU threads would only contend on the TRT execution context.

### First-run UX

`core.pre_check()` warns users when TRT EP is selected but no cached engines exist, setting the expectation that the first run will be slow (30–120 s) while compilation proceeds.

### Fallback

ONNX Runtime automatically falls back to the next provider in the list when TRT fails to build a subgraph (e.g. unsupported op).  The recommended provider chain is:

```
TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

Users who pass `--execution-provider tensorrt` get this chain automatically because `decode_execution_providers()` in `core.py` returns all matching providers from `onnxruntime.get_available_providers()`.

## Consequences

**Positive:**
- NVIDIA users get 2–5× FPS improvement with zero quality loss (same FP16 ONNX models).
- First-run compilation cost is paid once; subsequent startups are instant.
- No effect on non-NVIDIA platforms (macOS, AMD, CPU-only).
- Follows existing CoreML pattern: provider-specific options are encapsulated in `onnx_providers.py` and `tensorrt_cache.py`, not scattered across processor modules.

**Negative / Risks:**
- TensorRT compilation emits verbose logs that cannot easily be suppressed without `TF_CPP_MIN_LOG_LEVEL` equivalents.
- The TRT EP may silently fall back to CUDA EP if TensorRT is not installed correctly.  Users should verify "Applied providers" in startup logs.
- Engine files can be large (100s of MiB per model); users with constrained disk space may need to delete `models/trt_cache/` manually.
- Dynamic input shapes (batch size, resolution) may cause multiple engine compilations.  `trt_min_subgraph_size=3` mitigates unnecessary subgraph splits.

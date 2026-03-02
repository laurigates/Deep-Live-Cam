# ADR-0013: Apple Neural Engine Routing via CoreML Compute Units

## Status

Accepted

## Context

Deep-Live-Cam runs extended live webcam sessions on Apple Silicon hardware. The CoreML
Execution Provider in onnxruntime-silicon supports routing ONNX graph operations across
different hardware units via the `MLComputeUnits` option:

- `ALL` — ANE + GPU + CPU; CoreML decides optimal placement per op
- `CPUAndGPU` — GPU + CPU only; no ANE routing
- `CPUOnly` — CPU only; no hardware acceleration

Prior to this decision, the `CoreMLExecutionProvider` was configured without an explicit
`MLComputeUnits` option in the GFPGAN face enhancer (it used raw providers), while the
face swapper already set `MLComputeUnits=ALL` through `build_providers_config`. This
inconsistency meant the face enhancer bypassed CoreML provider configuration entirely on
Apple Silicon when using `CoreMLExecutionProvider`.

The Apple Neural Engine (ANE) is designed for sustained ML inference with lower power draw
than the GPU, and supports transformer-style attention blocks and convolutions well — the
same operations used by the inswapper and GFPGAN models.

## Decision

1. **Default to `MLComputeUnits=ALL`** across all ONNX sessions on Apple Silicon. This
   lets CoreML decide the optimal placement per operation, maximising ANE utilisation
   where beneficial while falling back to GPU/CPU as needed.

2. **Centralise provider configuration in `modules/onnx_providers.py`**. The
   `build_providers_config()` function is the single place that injects CoreML options.
   All ONNX session creators (`face_swapper`, `face_enhancer`, `_onnx_enhancer`) call
   this function — no module creates a session with raw provider strings on Apple Silicon.

3. **Accept `coreml_compute_units` as an injectable parameter** in `build_providers_config`
   with fallback to `modules.globals.coreml_compute_units`. This follows the same injectable
   pattern used for `execution_providers`, enabling tests to override without touching globals.

4. **Expose a `--coreml-compute-units` CLI flag** with choices `ALL`, `CPUAndGPU`,
   `CPUOnly`. This lets advanced users disable ANE routing for debugging, correctness
   verification, or latency profiling.

5. **Cache compiled CoreML models** via `ModelCacheDirectory` pointing to
   `~/.cache/deep-live-cam/coreml`. This avoids recompilation on each startup (CoreML
   compilation of the inswapper model takes 5-30 seconds on first use).

6. **Apple Silicon only**. The `IS_APPLE_SILICON` guard in `build_providers_config` ensures
   non-Apple platforms receive plain string provider names, with zero behaviour change.

## Valid CoreML Options (onnxruntime-silicon 1.24.2)

Per the project's cross-platform rules (`.claude/rules/cross-platform.md`), the only
validated options for `CoreMLExecutionProvider` in onnxruntime-silicon 1.24.2 are:

| Option | Value used | Notes |
|--------|-----------|-------|
| `ModelFormat` | `"MLProgram"` | Required for ANE compatibility |
| `MLComputeUnits` | `"ALL"` | Configurable via CLI flag |
| `SpecializationStrategy` | `"FastPrediction"` | Reduces first-run latency |
| `AllowLowPrecisionAccumulationOnGPU` | `1` | Matches fp16 model precision |
| `EnableOnSubgraphs` | `1` | Enables CoreML on subgraph nodes |
| `ModelCacheDirectory` | `~/.cache/…` | Avoids recompilation |

`RequireStaticShapes` and `MaximumCacheSize` are **not** valid in 1.24.2 and cause
silent fallback to CPU — they must not be used.

## Consequences

### Positive
- GFPGAN face enhancer now uses CoreML options on Apple Silicon (was using raw providers).
- Consistent provider configuration across all three ONNX model loaders.
- ANE routing may improve power efficiency during extended sessions.
- Model cache avoids recompilation on startup after first use.
- User can disable ANE routing for debugging via `--coreml-compute-units CPUOnly`.

### Negative / Risks
- ANE has lower precision on some operations; visual regression is possible (unlikely for
  these models, but unverified without hardware benchmarking).
- `MLComputeUnits=ALL` adds ~100-200ms first-inference latency as CoreML dispatches ops.
- The `models/coreml_cache/` directory grows as compiled models are cached; users must
  manage disk usage manually.

### Neutral
- No effect on non-Apple platforms (guard in `build_providers_config`).
- Face analyser uses InsightFace (separate session creation path) which already filters out
  CoreML due to dynamic shape incompatibility — no change needed there.

## References

- [CoreML EP docs — onnxruntime](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- `.claude/rules/gpu-acceleration.md` — CoreML provider option validation notes
- `.claude/rules/cross-platform.md` — Apple Silicon detection and CoreML guard patterns
- `modules/onnx_providers.py` — canonical implementation

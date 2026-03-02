# Feature: TensorRT Execution Provider

## Overview

Add `tensorrt` as a selectable execution provider for NVIDIA GPU users, delivering 2–5× FPS improvement over the existing `cuda` provider through TensorRT's graph fusion, kernel auto-tuning, and FP16 precision.  A persistent engine cache in `models/trt_cache/` eliminates the 30–120 s compilation cost after the first run.

## User Stories

- As an NVIDIA GPU user, I want to run `--execution-provider tensorrt` to get significantly higher FPS than `--execution-provider cuda`, so that face swapping feels smoother in live mode.
- As an NVIDIA GPU user, I want engine compilation to happen only on the first run and be cached afterwards, so that I don't wait 30–120 s every time I launch the application.
- As a macOS or CPU-only user, I want TensorRT to be completely invisible to me — my existing `coreml` or `cpu` provider must keep working exactly as before.
- As a user, I want a clear status message when TensorRT needs to compile engines for the first time, so I know why startup is slow.

## Acceptance Criteria

- [ ] `--execution-provider tensorrt` is accepted by the CLI and produces a `TensorrtExecutionProvider` session.
- [ ] Engine files are persisted to `models/trt_cache/` and reused on subsequent runs.
- [ ] A status message is displayed on first run explaining the compilation delay.
- [ ] If TensorRT compilation fails for a subgraph, ONNX Runtime falls back to CUDA EP automatically.
- [ ] `suggest_execution_threads()` returns `1` when TRT EP is active.
- [ ] FP16 precision is enabled (`trt_fp16_enable=1`) in all TRT sessions.
- [ ] The feature has no effect on macOS or CPU-only systems.
- [ ] Integration tests verify cache hit and cache miss paths.

## Non-Functional Requirements

### Performance

- ≥ 2× FPS improvement on `inswapper_128_fp16.onnx` compared to CUDA EP (measured over ≥ 300 frames).
- ≥ 3× FPS improvement on `gfpgan-1024.onnx` compared to CUDA EP.
- Second and subsequent runs must start within 5 s of normal CUDA EP startup time (i.e., no re-compilation).

### Resource Usage

- TRT engine cache occupies 100–500 MiB per model; document this in the README.
- Maximum GPU workspace: 2 GiB (configurable via `trt_max_workspace_size` option).

### Compatibility

- Works with `onnxruntime-gpu` ≥ 1.16.0 and a TensorRT installation on the system path.
- No effect on `onnxruntime-silicon` (macOS ARM) — that variant does not include TRT EP.
- Safe to import `modules.tensorrt_cache` on any platform (platform guard in the module).

## Out of Scope

- INT8 quantisation / calibration (future work if FP16 speedup is insufficient).
- Multi-GPU support (`device_id > 0`).
- TensorRT engine export / shipping pre-compiled engines in the repository.

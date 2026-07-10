# Feature: OSDFace Offline Face Enhancement

## Overview

Add OSDFace (CVPR 2025, one-step diffusion face restoration, ~978 M parameters) as an opt-in, **offline-only** face enhancer for image and video processing. OSDFace surpasses GFPGAN, CodeFormer, and DiffBIR in restoration quality but is far too slow for live mode (~100 ms/face on an NVIDIA A6000, ~1–2 s/frame on Apple Silicon MPS), so it is selectable only for headless/file processing and is explicitly blocked in the live webcam pipeline. It ships as a uv optional extra (`uv sync --extra osdface`) so the base install is unaffected. See [ADR 0015](../adrs/0015-osdface-one-step-diffusion-face-enhancement.md) for the integration strategy.

## User Stories

- As a video producer, I want to enhance swapped faces with `--frame-processor face_swapper face_enhancer_osdface` when processing a file, so that my offline output has the highest available restoration quality.
- As a live-mode user, I want a clear message when I try to enable OSDFace for the webcam, so that I understand it is offline-only and can pick GFPGAN/GPEN/CodeFormer instead — rather than the app silently ignoring my choice or freezing at 0.5 FPS.
- As a base-install user, I want OSDFace to be completely invisible to me — no extra downloads, no slower startup, no dependency changes — unless I explicitly run `uv sync --extra osdface`.
- As a user who selected OSDFace without installing the extra, I want an actionable error telling me exactly what command to run, so that I am not left with a raw `ImportError` traceback or a silently unenhanced output.

## Acceptance Criteria

- [ ] `--frame-processor face_swapper face_enhancer_osdface` is accepted by the CLI for image and video (headless) processing and runs the OSDFace enhancer after the swap. *(Note: issue #74 sketches a `--face-enhancer osdface` flag; no such flag exists in this repo — every enhancer is a `--frame-processor` choice, and OSDFace follows that existing convention.)*
- [ ] When `face_enhancer_osdface` is among the frame processors and a live webcam session starts, the session refuses to run OSDFace and publishes a clear status-bus message (e.g. "OSDFace is offline-only; use GFPGAN/GPEN/CodeFormer for live mode") — never a silent skip.
- [ ] OSDFace does **not** appear in the live-webcam enhancer UI (no checkbox, not in `_ENHANCER_NAMES`/`_ENHANCER_UI_KEYS`).
- [ ] If the `osdface` extra is not installed, selecting the processor fails fast with an actionable message instructing the user to run `uv sync --extra osdface` (ImportError caught at processor load; no traceback, no silent swap-without-enhancement).
- [ ] Model weights download on first use via `download_model_if_needed()` with primary + fallback URLs and sha256 checksums, into `models/`.
- [ ] Enhancement operates on 512×512 aligned face crops through the existing crop/alignment pipeline and pastes back seamlessly.
- [ ] Side-by-side comparison images vs GFPGAN and CodeFormer on the project test set are included in the implementation PR, demonstrating measurably better restoration quality.
- [ ] Base install (`uv sync` without extras) is byte-identical in dependencies and startup behavior before and after this feature.
- [ ] Works on CUDA (NVIDIA) and MPS (Apple Silicon); CPU fallback functions but warns about extreme slowness.

## Non-Functional Requirements

### Performance

- Offline throughput targets: ~100 ms/face on A6000-class NVIDIA GPUs; 1–2 s/frame on Apple Silicon MPS is acceptable for offline processing.
- Zero impact on live-mode FPS: the live pipeline must not import or load anything OSDFace-related.
- Zero impact on application startup time and base install size (lazy imports inside the processor module only).

### Resource Usage

- 4–6 GB VRAM during inference — documented in README; VRAM protected by the standard enhancer semaphore so multi-face frames do not exhaust memory.
- Multi-GB model download happens once, on first use, never at install or startup.

### Compatibility

- PyTorch-only inference (reuses the torch already in base dependencies); the extra group adds only base-compatible libraries (diffusers/transformers/accelerate/safetensors) — it must not downgrade base pins (`numpy>=1.23.5,<3`, `pillow>=12.0.0`).
- No ONNX Runtime involvement; unaffected by `--execution-provider` selection (device chosen as MPS → CUDA → CPU).

## Out of Scope

- Live webcam mode support (blocked by design; revisit only if per-face latency drops ~30×).
- ONNX or CoreML export of the OSDFace model (future research, tracked upstream).
- A UI toggle in the webcam enhancer panel.
- Installing the upstream OSDFace package directly (its old numpy/pillow pins conflict with base — see ADR 0015).

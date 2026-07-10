# ADR 0015: OSDFace One-Step Diffusion Face Enhancement (Offline-Only, PyTorch)

## Status
**Proposed** (Jul 2026, issue #74 — no implementation exists yet)

## Context

Deep-Live-Cam currently offers four face enhancers, all real-time capable:
GFPGAN (PyTorch), GPEN-256/GPEN-512 and CodeFormer (ONNX, via
`_onnx_enhancer_factory`). All of them are GAN- or codebook-based restorers.

**OSDFace** (CVPR 2025, [jkwang28/OSDFace](https://github.com/jkwang28/OSDFace),
[arXiv:2411.17163](https://arxiv.org/abs/2411.17163)) is a one-step diffusion
face restorer built on a Stable Diffusion 2.1 UNet with a visual-representation
(VQ) face embedder. Its output quality surpasses GFPGAN, CodeFormer, and
DiffBIR on standard blind face restoration benchmarks. Key facts that shape
this decision:

- **Model size**: ~978 M parameters (SD 2.1 UNet + VQ embedder + VAE).
- **Latency**: ~100 ms/face on an NVIDIA A6000; ~1–2 s/frame on Apple Silicon
  MPS. Far too slow for the 30–60 FPS live webcam pipeline — viable only for
  offline image/video processing.
- **VRAM**: 4–6 GB during inference.
- **Runtime**: PyTorch-only inference (diffusers-based). There is **no
  ONNX/CoreML export path today** — the iterative-free single step still relies
  on diffusers scheduler plumbing and the VQ embedder; ONNX export of the
  VQ-encoder is listed as future research, not a current option. This breaks
  with our dual-runtime convention (ADR 0006) where ONNX handles anything
  latency-critical.
- **Dependency conflicts**: the upstream OSDFace repository pins old `numpy`
  and `pillow` versions that conflict with our base dependencies
  (`pyproject.toml` pins `numpy>=1.23.5,<3` and `pillow>=12.0.0`). Installing
  the upstream package as-is would downgrade shared libraries used by every
  other processor.

The base install already ships PyTorch (GFPGAN needs it — ADR 0006), so the
marginal dependency cost of OSDFace is `diffusers`/`transformers` and the model
weights, not a new runtime.

The question is **how to integrate OSDFace without destabilizing the base
install**, given the size, the PyTorch-only constraint, and the conflicting
upstream pins.

## Decision

1. **Offline-only processor.** OSDFace is exposed as a new frame-processor
   plugin (`face_enhancer_osdface`, per ADR 0002) selectable **only for image
   and video (headless/offline) processing**. Live webcam mode explicitly
   rejects it with a clear status-bus message; it is excluded from the webcam
   enhancer UI entirely.

2. **PyTorch-only inference, no ONNX/CoreML path.** Device selection is
   MPS (Apple Silicon) → CUDA (NVIDIA) → CPU (discouraged, warn about
   multi-minute frame times). We do not attempt ONNX export; if a viable
   export appears upstream, it becomes a new ADR.

3. **Dependency isolation via a uv optional extra group** —
   `[project.optional-dependencies] osdface` in `pyproject.toml`, installed
   with `uv sync --extra osdface`:
   - We **do not install the upstream OSDFace package** (that is what carries
     the old numpy/pillow pins). Instead the extra group pins only
     base-compatible libraries (`diffusers`, `transformers`, `accelerate`,
     `safetensors`), and the inference path is vendored/re-implemented inside
     `modules/processors/frame/face_enhancer_osdface.py` against those
     libraries.
   - All OSDFace imports are **lazy** (inside functions, guarded by
     `try/except ImportError`), so the base install never pays any import or
     startup cost, and users without the extra get an actionable
     "run `uv sync --extra osdface`" error instead of a stack trace.

   **Rejected alternative — fully separate venv + subprocess**: run OSDFace in
   its own virtual environment with upstream pins intact, communicating over a
   subprocess pipe. This guarantees zero dependency interference but adds a
   second environment to manage, serializes frames across a process boundary
   (significant overhead at 512×512 crops), complicates error reporting, and
   `subprocess`/`fork` is a known hazard on macOS in this codebase (see
   `.claude/rules/cross-platform.md`). It remains the **documented fallback**
   if the vendored-inference approach hits an irreconcilable pin (e.g. a
   diffusers release that itself requires an incompatible numpy).

4. **Model weights** are downloaded on first use via
   `utilities.download_model_if_needed()` with primary + fallback URLs and
   sha256 checksums, stored in `models/` (ADR 0007) — never bundled.

## Consequences

### Positive
✓ **Best-in-class quality** for offline restoration — surpasses GFPGAN,
  CodeFormer, and DiffBIR
✓ **Zero impact on base install**: optional extra + lazy imports mean no new
  startup cost, download, or dependency for users who don't opt in
✓ **No new runtime**: reuses the PyTorch already required by GFPGAN (ADR 0006)
✓ **Plugin architecture preserved**: standard five-method frame-processor
  interface, no special-casing in the pipeline core (ADR 0002)
✓ **Base pins untouched**: numpy/pillow stay current; vendoring sidesteps the
  upstream conflict entirely

### Negative
✗ **First processor with a hard live-mode exclusion** — needs an explicit
  guard and user messaging; a silent failure here would look like a hang
✗ **Vendored inference code must track upstream** — bug fixes in
  jkwang28/OSDFace don't arrive automatically
✗ **~978 M-param model**: multi-GB download, 4–6 GB VRAM requirement excludes
  low-end GPUs
✗ **Cannot reuse `_onnx_enhancer_factory`** — a standalone PyTorch module
  duplicates some crop/paste plumbing
✗ **diffusers/transformers in the extra group** enlarge the opt-in install by
  hundreds of MB

### Mitigations
- Live-mode guard publishes a clear message via `modules.status_bus.BUS`
  ("OSDFace is offline-only…") instead of failing silently
- Vendored inference kept minimal (single-step UNet call + VQ embed), pinned
  to a named upstream commit in a header comment for diffability
- VRAM protected by the existing `THREAD_SEMAPHORE` enhancer pattern
  (`.claude/rules/face-processing.md`); documented 4–6 GB requirement in README
- Separate-venv fallback documented (above) if pins become irreconcilable

## Evidence

- Issue #74 — integration request with benchmark numbers
- [OSDFace paper (arXiv:2411.17163)](https://arxiv.org/abs/2411.17163) — CVPR
  2025; quality comparisons vs GFPGAN/CodeFormer/DiffBIR
- [jkwang28/OSDFace](https://github.com/jkwang28/OSDFace) — upstream reference
  implementation and requirement pins
- `pyproject.toml` base pins: `numpy>=1.23.5,<3`, `pillow>=12.0.0`, torch
  already present on all platforms

## Related Decisions
- [ADR 0002: Plugin Architecture for Frame Processors](0002-plugin-architecture-for-frame-processors.md)
- [ADR 0006: Dual-Runtime Approach (PyTorch + ONNX)](0006-dual-runtime-pytorch-onnx-separation.md) — precedent for PyTorch-side enhancement
- [ADR 0007: Model Fallback and Switching Mechanism](0007-model-fallback-and-switching-mechanism.md) — download with fallback URLs
- [ADR 0011: ONNX-Based Face Enhancement (GPEN/BFR)](0011-onnx-based-face-enhancement-gpen-bfr.md) — the ONNX enhancer family OSDFace deliberately does not join

## Future Improvements
- ONNX/CoreML export of the VQ embedder and single-step UNet (upstream research)
- fp16/int8 quantization to cut VRAM below 4 GB
- Batch-of-faces inference for multi-face frames
- Revisit live-mode eligibility if per-face latency drops below ~30 ms

**Last Reviewed**: Jul 10, 2026 | **Confidence**: Medium (proposed; no implementation yet)

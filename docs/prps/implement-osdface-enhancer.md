# PRP: Implement OSDFace Offline Face Enhancer

## Objective

Integrate OSDFace (CVPR 2025 one-step diffusion face restoration, ~978 M params, PyTorch-only) as a new offline-only frame processor `face_enhancer_osdface`, installable via an optional `osdface` extra, blocked in live webcam mode, with model download/fallback and a graceful error when the extra is missing.

## Background

See: [ADR 0015](../adrs/0015-osdface-one-step-diffusion-face-enhancement.md) | [PRD](../prds/osdface-offline-face-enhancement.md)

Key constraints established there:

- **PyTorch-only** — no ONNX/CoreML path exists. The processor **must NOT** use `modules/processors/frame/_onnx_enhancer_factory.py` (that factory builds ONNX `InferenceSession`s; the CodeFormer one-liner pattern does not apply here). OSDFace needs a standalone module.
- **Offline-only** — image/video processing only; live mode rejects it with a status-bus message.
- **Dependency isolation** — upstream OSDFace pins old numpy/pillow that conflict with base (`numpy>=1.23.5,<3`, `pillow>=12.0.0`). Do not install the upstream package; vendor the inference against base-compatible `diffusers`/`transformers` pins in an optional extra group. Fallback if pins prove irreconcilable: separate venv + subprocess (documented in ADR 0015, not implemented here).
- **CLI convention** — the issue's `--face-enhancer osdface` spelling is normalized to the repo convention: `--frame-processor face_enhancer_osdface` (all enhancers are `--frame-processor` choices in `modules/core.py`).

Performance envelope (issue #74): ~100 ms/face on A6000, ~1–2 s/frame on Apple Silicon MPS, 4–6 GB VRAM. References: <https://github.com/jkwang28/OSDFace>, <https://arxiv.org/abs/2411.17163>.

## Implementation Steps

### Step 1: Optional dependency group

**`pyproject.toml`** — add a new `[project.optional-dependencies]` table (none exists today; place it after the `dependencies` list, before `[build-system]`):

```toml
[project.optional-dependencies]
osdface = [
    "diffusers>=0.31,<1",
    "transformers>=4.45,<5",
    "accelerate>=1.0,<2",
    "safetensors>=0.4",
]
```

Pins must be verified against the base lock at implementation time: `uv sync --extra osdface` must succeed **without changing any base pin** (numpy, pillow, torch). Do not add the upstream `osdface` package. Torch is already a base dependency — do not duplicate it in the extra.

### Step 2: New processor module

**`modules/processors/frame/face_enhancer_osdface.py`** (new) — standalone module implementing all five methods of `FRAME_PROCESSORS_INTERFACE` (`modules/processors/frame/core.py`): `pre_check`, `pre_start`, `process_frame`, `process_image`, `process_video`.

- `NAME = "DLC.FACE-ENHANCER-OSDFACE"`
- **Lazy imports**: no top-level `torch`/`diffusers` imports. Mirror the `_ensure_torch()` pattern in `modules/core.py` — a module-level `_ensure_osdface_deps() -> bool` that imports `torch` + `diffusers`/`transformers` on first call inside `try/except ImportError`, caches the tri-state result, and on failure publishes an actionable message: `"OSDFace dependencies not installed. Run: uv sync --extra osdface"`. `pre_check()` returns `False` in that case so the pipeline fails fast (no silent swap-without-enhancement).
- **Singleton loader**: `OSDFACE_ENHANCER = None` module global + `get_osdface_enhancer(device: str | None = None)` protected by a `threading.Lock()`, following the injectable-provider pattern from `.claude/rules/face-processing.md` — accept an explicit `device` parameter for tests; derive from torch at runtime otherwise (`mps` if `torch.backends.mps.is_available()`, else `cuda` if `torch.cuda.is_available()`, else `cpu` with a slowness warning).
- **VRAM protection**: wrap each enhancement call in `THREAD_SEMAPHORE = threading.Semaphore(1)` (model needs 4–6 GB; concurrency of 1).
- **Inference**: vendored single-step pipeline (VQ face embed → one UNet step → VAE decode) on 512×512 aligned crops, reusing the existing crop/align/paste helpers used by the other enhancers. Pin the referenced upstream commit of jkwang28/OSDFace in a header comment.
- `process_frame` must return the input frame unchanged when no face is detected (never `None`).
- **Status updates** go through `modules.status_bus.BUS` / `modules.core.update_status` — never `import modules.ui`.

### Step 3: Model download

In `pre_check()`, use `modules/utilities.py:download_model_if_needed(model_file, model_urls, NAME)` (models land in `modules.paths.MODELS_DIR`) with:

- primary URL (Hugging Face release of the OSDFace weights) + fallback mirror URL, per ADR 0007 and `.claude/rules/face-processing.md` "Model Download and Caching";
- sha256 checksums passed through to `conditional_download(..., expected_checksums=...)`;
- weights fetched on first use only — never at import or install time.

If `download_model_if_needed` needs no change this step touches only the new module; extend `modules/utilities.py` only if checksum plumbing is missing for multi-file weights.

### Step 4: Globals + CLI registration

**`modules/globals.py`** — add to the `fp_ui` dict:

```python
"face_enhancer_osdface": False,
```

**`modules/core.py`**:

1. Add `"face_enhancer_osdface"` to the `--frame-processor` `choices` list.
2. Add `"face_enhancer_osdface"` to the `enhancer_key` tuple in the "ENHANCER tumblers" loop so `fp_ui` is synced from CLI args.
3. Headless validation: when `face_enhancer_osdface` is among the frame processors but the run is **not** headless (no `-s/-t/-o`), do not launch the GUI with it active — strip it and `update_status` the offline-only message (belt-and-braces with Step 5's live-mode guard).

### Step 5: Live-mode guard

**`modules/ui_webcam.py`**:

- Deliberately **EXCLUDE** `"DLC.FACE-ENHANCER-OSDFACE"` from `_ENHANCER_NAMES` and `_ENHANCER_UI_KEYS` — it must never gain a webcam UI toggle or enter the async enhancement thread.
- At webcam session start (where `get_frame_processors_modules(config.frame_processors)` is called in the processing thread setup), if `"face_enhancer_osdface"` appears in `config.frame_processors`, filter it out of the session's processor list and publish once via `modules.status_bus.BUS`:
  `"OSDFace is offline-only; use GFPGAN/GPEN/CodeFormer for live mode"`.
  Never import `modules.ui` from this path; never silently drop it without the message.

### Step 6: Documentation touch-ups

- README: document `uv sync --extra osdface`, the 4–6 GB VRAM requirement, and the offline-only restriction.
- Update `docs/blueprint/feature-tracker.json` FR-011 status as milestones land.

## Success Criteria

- [ ] `uv sync` (no extras) resolves to the identical lock as before the change; `uv sync --extra osdface` succeeds without downgrading numpy/pillow/torch.
- [ ] `uv run run.py -s s.jpg -t t.mp4 -o o.mp4 --frame-processor face_swapper face_enhancer_osdface` runs end-to-end on CUDA and MPS hardware.
- [ ] Without the extra installed, the same command exits fast with the `uv sync --extra osdface` message (no traceback, no unenhanced output silently written).
- [ ] Starting a live webcam session with `face_enhancer_osdface` selected publishes the offline-only message and runs the session without OSDFace.
- [ ] Side-by-side comparison images vs GFPGAN/CodeFormer included in the PR.
- [ ] `uv run pytest -x -q` green; new `tests/test_face_enhancer_osdface.py` passes.
- [ ] No live-mode FPS or startup-time regression (15 min live test per CLAUDE.md).

## Testing Strategy

Per `.claude/rules/testing.md` — TDD (failing test first), all heavy deps mocked; GPU-dependent paths never execute real inference in CI.

**New `tests/test_face_enhancer_osdface.py`** (model on `tests/test_face_enhancer_codeformer.py` and `tests/test_download_model_if_needed.py`):

1. **Interface compliance** — module exposes all five `FRAME_PROCESSORS_INTERFACE` methods and `NAME == "DLC.FACE-ENHANCER-OSDFACE"`; loadable via `load_frame_processor_module("face_enhancer_osdface")`.
2. **Graceful ImportError** — patch the lazy import to raise `ImportError` at the module import path (never `patch.object(..., '__init__')`); assert `pre_check()` returns `False` and the `uv sync --extra osdface` message is published to the status bus.
3. **Singleton reset discipline** — save/restore `OSDFACE_ENHANCER` around tests exactly like the `FACE_SWAPPER` pattern in `.claude/rules/testing.md`; pass `device` explicitly, never set globals.
4. **No-face pass-through** — `process_frame` returns the input frame unmodified when the analyser yields no faces.
5. **Model download** — mock `download_model_if_needed`; assert primary+fallback URLs and checksums are passed, and `pre_check()` fails when it returns `False`.
6. **CLI registration** — `face_enhancer_osdface` accepted by the `--frame-processor` choices; `fp_ui["face_enhancer_osdface"]` synced from args.
7. **Live-mode guard** — with `face_enhancer_osdface` in `config.frame_processors`, the webcam processing setup filters it out and publishes the offline-only message via a subscribed test callback on `BUS`; assert it is absent from `_ENHANCER_NAMES`/`_ENHANCER_UI_KEYS`.
8. **Lazy-import hygiene** — importing `modules.processors.frame.face_enhancer_osdface` with `diffusers` absent from `sys.modules` must not raise and must not import torch at module level.

**GPU benchmarking** (requires real hardware — out of container/CI scope, run manually before merge): 300-frame average FPS protocol from `.claude/rules/gpu-acceleration.md` on A6000-class CUDA and Apple Silicon MPS; record ms/face and peak VRAM in the PR; confirm zero live-mode FPS change with the feature merged but not selected.

## Known Risks

1. **diffusers/transformers pin drift** — a future diffusers release may require numpy/pillow versions conflicting with base. Mitigation: upper-bound pins in the extra; escalate to the separate-venv fallback (ADR 0015) only if irreconcilable.
2. **Weights hosting stability** — upstream may move the checkpoint. Mitigation: primary + fallback URLs with checksums (ADR 0007 pattern).
3. **MPS operator gaps** — some diffusers ops historically fell back to CPU on MPS. Mitigation: verify the 1–2 s/frame target on real Apple Silicon before merge; document any `PYTORCH_ENABLE_MPS_FALLBACK` requirement.
4. **Vendored code divergence** — upstream fixes don't auto-propagate. Mitigation: pin the vendored-from commit in a header comment; keep the vendored surface minimal.

## File Summary

| File | Change Type | Description |
|------|------------|-------------|
| `modules/processors/frame/face_enhancer_osdface.py` | Create | Standalone PyTorch processor (five-method interface, lazy imports, semaphore) |
| `pyproject.toml` | Modify | New `[project.optional-dependencies]` `osdface` group |
| `modules/utilities.py` | Modify (if needed) | Checksum plumbing for multi-file weights via `download_model_if_needed` |
| `modules/globals.py` | Modify | Add `face_enhancer_osdface` to `fp_ui` |
| `modules/core.py` | Modify | `--frame-processor` choice + enhancer tumbler + headless validation |
| `modules/ui_webcam.py` | Modify | Live-mode guard (filter + BUS message); keep out of `_ENHANCER_NAMES`/`_ENHANCER_UI_KEYS` |
| `tests/test_face_enhancer_osdface.py` | Create | Unit tests (interface, ImportError path, guard, CLI) |
| `README.md` | Modify | Extra install command, VRAM requirement, offline-only note |

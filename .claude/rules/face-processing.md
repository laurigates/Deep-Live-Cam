# Face Processing Patterns

Derived from recurring fix commits: None embeddings crash (PR #980), source_target_map typo
(#1565), face mapper bugs in live mode (#572, #598), and face enhancer device errors (#829).

## Face Detection

- Always guard against `None` face results before accessing embeddings or bounding boxes
  (source: "Fix face swapping crash due to None face embeddings")
- Log a clear message when no faces are found — do not silently skip frames
- Return the unmodified input frame when no face is detected; never return `None`

## InsightFace Singleton Behavior

- `FaceAnalysis.prepare()` sets `det_size` **only on the first call** — subsequent calls are
  silently ignored with `"warning: det_size is already set in detection model, ignore"`
- To change `det_size` at runtime (e.g., switching between live mode and video mode), you must
  **recreate the `FaceAnalysis` instance entirely** — calling `prepare()` again is a no-op
- Protect the singleton with a lock when recreating it from multiple threads

## Embedding Validation

- Validate face embedding shape and dtype before passing to ONNX swap model
- A mismatched embedding silently produces garbage output — fail fast with a descriptive error

## Face Mapping (FaceMapStore)

Face map state is owned by `modules.face_map_store.STORE` (`FaceMapStore` singleton).
`source_target_map`, `simple_map`, and `MAP_LOCK` no longer exist in `modules.globals`.

- `from modules.face_map_store import STORE as _MAP_STORE`
- `_MAP_STORE.get_entries()` — always returns a **snapshot** (copy), never the live list
- `_MAP_STORE.set_entries(new_list)` — atomic replacement
- `_MAP_STORE.add_blank()`, `_MAP_STORE.clear()`, `_MAP_STORE.has_valid_map()`
- `_MAP_STORE.simplify()` — builds simple map from paired entries in-place
- `_MAP_STORE.get_simple_map()` — snapshot copy of the simplified map

`FaceMapStore` uses `threading.Lock()` (not `RLock`) — if you need re-entrance it is
a design error.  All map read operations return snapshots to avoid holding the lock
while processing.

- Keep map entries as `{source_face, target_face}` dicts; avoid positional indexing
- When live-mode faces exceed available map entries, fall back to the first map entry rather
  than crashing (evidence: PR #572)

## Frame Processor Pipeline

- Processors run sequentially: `face_swapper` → `face_enhancer` → `face_masking`
- Each processor receives the output of the previous one as its input frame
- A processor must return a valid frame even if it performs no operation (pass-through)
- Implement all five interface methods: `pre_check`, `pre_start`, `process_frame`,
  `process_image`, `process_video`

## Face Enhancer (GFPGAN)

- Load the model conditionally based on the active device — do not load on CPU if GPU is
  available and vice versa (evidence: "Make Face Enhancer Model device Conditional")
- Wrap enhancement in a semaphore to prevent concurrent VRAM exhaustion on multi-face frames
- Enhancement is optional; the pipeline must work correctly with it disabled

## Async Enhancement Thread

Enhancement runs on a dedicated thread (`_enhancement_thread_func`) decoupled from the
swap+masking processing thread — do not move enhancement back into the processing loop.

- **Submit protocol**: processing thread writes `{frame, faces, map_faces, processor, seq}` to
  `enhancement_input[0]` under `enhancement_lock`; seq is a monotonically incrementing int
- **Read protocol**: processing thread reads `enhancement_output[0]` and compares `seq` against
  `last_consumed_enh_seq`; only update `latest_enhanced_frame` when seq is new
- **Skip-frame split**: on skip frames, suppress *submission* but still read and display the
  latest output — this keeps display smooth during sparse inference
- **Toggle-off cleanup**: when the enhancer is disabled, clear all shared state:
  `latest_enhanced_frame = None`, `enhancement_input[0] = None`, `enhancement_output[0] = None`
- **First-frame latency**: the first ~100-200ms after enabling enhancement, frames display
  swapped-but-unenhanced — this is expected and not a bug
- Use `process_frame(None, frame, faces=faces)` for normal mode,
  `process_frame_v2(frame)` for map_faces mode

## Mouth Masking

- Use face landmarks for mouth region coordinates; do not use fixed pixel offsets
- Apply Poisson blending (`cv2.seamlessClone`) at the mask boundary for smooth transitions
  (added in v2.0.2c; do not revert to hard alpha compositing)
- Test with both `--mouth-mask` enabled and disabled before committing mouth-mask changes

## Injectable Provider Pattern

All three ONNX model loaders accept an explicit providers list for testing and
injection — fall back to `modules.globals.execution_providers` when omitted:

| Loader | Injection parameter |
|--------|---------------------|
| `get_face_analyser()` | `config: Optional[ProcessingConfig]` → `config.execution_providers` |
| `get_face_swapper()` | `providers: list \| None` |
| `get_face_enhancer()` | `providers: list \| None` |

Never set `modules.globals.execution_providers` in tests — pass providers explicitly.

## Status Updates

Do **not** import `modules.ui` in non-UI modules.  Publish status messages via:

```python
from modules.status_bus import BUS
BUS.publish("Processing...", "MY_MODULE_NAME")
```

Or use `from modules.core import update_status` — it routes through `BUS` automatically.
GUI mode subscribes `ui.update_status` to `BUS` during startup; headless mode gets the
`print()` fallback in `update_status`.

**Private constant cross-module import = wrong abstraction boundary.**
`from module import _PRIVATE_CONST` across module boundaries is a code smell.
Expose such values as public class attributes instead (e.g., `_LIVE_DET_SIZE` became
`FaceAnalyser.LIVE_DET_SIZE`).

## Model Download and Caching

- Use `utilities.conditional_download()` for all model files — never download directly in
  processor code
- Support fallback sources: if the primary URL fails, try the secondary URL before surfacing
  an error to the user (evidence: "Creating a fallback and switching of models", Aug 2025)
- Store models in the `models/` directory; do not use temp directories for model files

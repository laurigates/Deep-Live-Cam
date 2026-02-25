# PRP: DRY Refactoring Plan for Deep-Live-Cam

## Objective

Eliminate all identified code duplication across the Deep-Live-Cam codebase through
incremental, test-driven phases. Each phase is self-contained and mergeable
independently. Phases are ordered to satisfy dependencies and maximize impact while
minimizing risk.

**Estimated total savings**: ~500+ lines of duplicated code eliminated.

---

## Execution Order

Phases are ordered so that each phase's dependencies are satisfied before it begins:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9
```

Dependency graph:
```
Phase 1 (platform_info)  ←── Phase 2 (onnx_providers) uses IS_APPLE_SILICON
Phase 1 (platform_info)  ←── Phase 7 (camera) uses IS_WINDOWS, IS_APPLE_SILICON
Phase 3 (factory) depends on Phase 1 and Phase 2
Phase 4 (pre_check) used by Phase 3 factory
Phase 5 (detect_faces) used by Phase 3 factory
Phase 6 (process_frame_v2) subsumed by Phase 3 factory
```

---

## Phase 1: Extract Centralized Platform Detection Constants

**Priority**: HIGH | **Risk**: Very Low | **Lines saved**: ~4 (but eliminates inconsistency)

### Motivation

The Apple Silicon detection check is defined in 4 separate locations with 3 different
formulations — some use `platform.system()`, others `sys.platform`. Having different
variable names and detection methods is a maintenance hazard.

| File | Variable | Method |
|------|----------|--------|
| `face_swapper.py:44` | `IS_APPLE_SILICON` | `platform.system() == 'Darwin'` |
| `_onnx_enhancer.py:18` | `IS_APPLE_SILICON` | `platform.system() == "Darwin"` |
| `face_analyser.py:20` | `_IS_APPLE_SILICON` | `platform.system() == 'Darwin'` |
| `gpu_processing.py:34` | `_ON_MACOS_ARM` | `sys.platform == "darwin"` |

### Changes

**Create** `modules/platform_info.py`:
```python
"""Platform detection constants — single source of truth."""
import platform
import sys

IS_APPLE_SILICON: bool = sys.platform == "darwin" and platform.machine() == "arm64"
IS_WINDOWS: bool = sys.platform == "win32"
IS_LINUX: bool = sys.platform == "linux"
```

**Modify** each consuming module — replace local definitions with:
```python
from modules.platform_info import IS_APPLE_SILICON
```

For `gpu_processing.py`, replace `_ON_MACOS_ARM` with `IS_APPLE_SILICON`.
For `face_analyser.py`, replace `_IS_APPLE_SILICON` with `IS_APPLE_SILICON`.

### Backward Compatibility

All replaced variables are module-internal (private convention). No public API changes.

### Tests

Create `tests/test_platform_info.py`:
- Verify `IS_APPLE_SILICON` is a bool
- Mock `sys.platform` and `platform.machine()` to test detection logic
- Verify `IS_WINDOWS` and `IS_LINUX` correctness

---

## Phase 2: Extract CoreML Provider Configuration

**Priority**: HIGH | **Risk**: Low | **Lines saved**: ~30

### Motivation

The CoreML EP configuration block (cache directory creation, options dict with
`ModelFormat`, `MLComputeUnits`, `SpecializationStrategy`, etc.) is duplicated verbatim:
- `face_swapper.py:109-126` (in `get_face_swapper()`)
- `_onnx_enhancer.py:27-47` (in `create_onnx_session()`)

If a CoreML option needs to change (as happened when `RequireStaticShapes` was found
invalid), both locations must be updated in sync.

### Changes

**Create** `modules/onnx_providers.py`:
```python
"""Build ONNX Runtime provider configuration lists."""
import os
from typing import List, Union, Tuple
from modules.platform_info import IS_APPLE_SILICON

ProviderConfig = Union[str, Tuple[str, dict]]

_COREML_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "deep-live-cam", "coreml"
)

def build_providers_config(providers: List[str]) -> List[ProviderConfig]:
    """Convert a flat list of provider names into a config list with options."""
    config: List[ProviderConfig] = []
    for p in providers:
        if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            os.makedirs(_COREML_CACHE_DIR, exist_ok=True)
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                    "EnableOnSubgraphs": 1,
                    "MaximumCacheSize": 1024 * 1024 * 512,
                    "ModelCacheDirectory": _COREML_CACHE_DIR,
                },
            ))
        else:
            config.append(p)
    return config
```

**Modify** `_onnx_enhancer.py:create_onnx_session()` — replace inline config building:
```python
from modules.onnx_providers import build_providers_config

def create_onnx_session(model_path):
    providers_config = build_providers_config(modules.globals.execution_providers)
    return onnxruntime.InferenceSession(model_path, providers=providers_config)
```

**Modify** `face_swapper.py:get_face_swapper()` — replace inline config block:
```python
from modules.onnx_providers import build_providers_config
# ...
providers_config = build_providers_config(modules.globals.execution_providers)
FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers_config)
```

### Tests

Create `tests/test_onnx_providers.py`:
- Non-CoreML providers pass through as strings
- CoreMLExecutionProvider on Apple Silicon becomes tuple with expected options
- Cache directory is created
- Options dict contains all expected keys (regression test)

---

## Phase 3: Collapse Three ONNX Enhancer Modules into a Factory

**Priority**: CRITICAL | **Risk**: Low | **Lines saved**: ~290

### Motivation

`face_enhancer_gpen256.py` (135 lines), `face_enhancer_gpen512.py` (135 lines), and
`face_enhancer_codeformer.py` (147 lines) are near-identical. They share all 11 functions
— differing only in 4 constants and CodeFormer's extra fidelity input. This is ~370
lines of pure duplication.

### Changes

**Create** `modules/processors/frame/_onnx_enhancer_factory.py`:

A factory function `create_onnx_enhancer_module(name, input_size, model_url, model_file, extra_input_fn=None)` that returns a dict of all required module-level functions. Uses closures to parameterize the shared logic:

```python
def create_onnx_enhancer_module(name, input_size, model_url, model_file, extra_input_fn=None):
    """Create a complete set of frame processor functions for an ONNX enhancer.

    Returns a dict suitable for globals().update() — the caller module gets all
    required plugin-interface functions as module-level attributes.
    """
    _model = ModelHolder()
    _load_error_logged_holder = [False]  # mutable container for closure

    def _load_model():
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(...)
        session = create_onnx_session(model_path)
        return session

    def _warmup(session):
        warmup_session(session)

    def get_enhancer():
        return _model.get(loader_fn=_load_model, warmup_fn=_warmup)

    def pre_check():
        return download_model_if_needed(model_file, [model_url], name)

    def pre_start():
        if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
            update_status("Select an image or video for target path.", name)
            return False
        return True

    def enhance_face(temp_frame, face):
        # ... get session, call enhance_face_onnx with optional extra_inputs ...

    def process_frame(source_face, temp_frame, faces=None, live_mode=False):
        if faces is None:
            faces = detect_faces(temp_frame)
        for face in faces:
            if face is not None:
                temp_frame = enhance_face(temp_frame, face)
        return temp_frame

    def process_frame_v2(temp_frame, faces=None, live_mode=False):
        return process_frame(None, temp_frame, faces=faces, live_mode=live_mode)

    # ... process_frames, process_image, process_video ...

    return {
        'NAME': name, 'INPUT_SIZE': input_size,
        'pre_check': pre_check, 'pre_start': pre_start,
        'enhance_face': enhance_face, 'get_enhancer': get_enhancer,
        'process_frame': process_frame, 'process_frames': process_frames,
        'process_image': process_image, 'process_video': process_video,
        'process_frame_v2': process_frame_v2,
    }
```

**Reduce** each thin module to ~15-25 lines:

```python
# face_enhancer_gpen256.py
from modules.processors.frame._onnx_enhancer_factory import create_onnx_enhancer_module

_ns = create_onnx_enhancer_module(
    name="DLC.FACE-ENHANCER-GPEN256",
    input_size=256,
    model_url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx",
    model_file="GPEN-BFR-256.onnx",
)
globals().update(_ns)
```

```python
# face_enhancer_codeformer.py (with fidelity support)
import numpy as np
import modules.globals
from modules.processors.frame._onnx_enhancer_factory import create_onnx_enhancer_module

DEFAULT_FIDELITY = 0.7

def _get_fidelity():
    return getattr(modules.globals, "codeformer_fidelity", DEFAULT_FIDELITY)

_ns = create_onnx_enhancer_module(
    name="DLC.FACE-ENHANCER-CODEFORMER",
    input_size=512,
    model_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
    model_file="codeformer.onnx",
    extra_input_fn=lambda: {"w": np.array([_get_fidelity()], dtype=np.float32)},
)
globals().update(_ns)
```

### Backward Compatibility

The plugin loader (`modules/processors/frame/core.py`) uses `importlib.import_module`
and `hasattr(module, method_name)`. Using `globals().update(_ns)` places all required
functions as module-level attributes, so the dynamic loader sees them identically. The
`NAME` constant is also set at module level.

### Tests

Create `tests/test_onnx_enhancer_factory.py`:
- `create_onnx_enhancer_module` returns dict with all required keys
- Each returned function is callable
- `NAME` and `INPUT_SIZE` match what was passed in
- `enhance_face` passes `extra_inputs` when `extra_input_fn` is provided
- `process_frame(None, frame)` and `process_frame_v2(frame)` produce identical results

---

## Phase 4: Extract Pre-Check Model Download Pattern

**Priority**: MEDIUM | **Risk**: Low | **Lines saved**: ~40

### Motivation

Every processor's `pre_check()` repeats the same download-verify pattern with only
`MODEL_FILE`, `MODEL_URL`, `MODELS_DIR` varying.

### Changes

**Add** to `modules/utilities.py`:
```python
def download_model_if_needed(
    model_file: str,
    model_urls: list[str],
    processor_name: str,
    models_dir: str | None = None,
) -> bool:
    """Download a model if not present. Returns True if model is available."""
    if models_dir is None:
        from modules.paths import MODELS_DIR
        models_dir = MODELS_DIR
    model_path = os.path.join(models_dir, model_file)
    if not os.path.exists(model_path):
        update_status(f"Downloading {model_file}...", processor_name)
    conditional_download(models_dir, model_urls)
    if not os.path.exists(model_path):
        update_status(
            f"Model not found at {model_path}. Download may have failed.",
            processor_name,
        )
        return False
    return True
```

**Modify** each processor's `pre_check()` to become a one-liner:
```python
def pre_check() -> bool:
    return download_model_if_needed(MODEL_FILE, [MODEL_URL], NAME)
```

Phase 3's factory uses this helper directly, so the three ONNX enhancers get it for free.

### Tests

- Mock `os.path.exists` and `conditional_download`
- Verify returns `True` when model exists after download
- Verify returns `False` when model missing after attempt
- Verify `update_status` called with correct format string

---

## Phase 5: Extract Face Detection Conditional

**Priority**: LOW | **Risk**: Low | **Lines saved**: ~12

### Motivation

This pattern appears in 6+ locations:
```python
faces = get_many_faces(frame) if modules.globals.many_faces else [get_one_face(frame)]
```

### Changes

**Add** to `modules/face_analyser.py`:
```python
def detect_faces(frame: Frame) -> list:
    """Return detected faces based on the current many_faces setting."""
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
        return faces if faces else []
    else:
        face = get_one_face(frame)
        return [face] if face is not None else []
```

This also fixes the subtle issue where `[get_one_face(frame)]` can produce `[None]`,
which requires `if face is not None` guards downstream.

Phase 3's factory uses `detect_faces()` directly.

### Tests

- Mock `many_faces=True`, verify all faces returned
- Mock `many_faces=False`, verify single face in list
- Verify empty list when no face detected (not `[None]`)

---

## Phase 6: Eliminate `process_frame_v2` Duplication

**Priority**: MEDIUM | **Risk**: Very Low | **Lines saved**: (included in Phase 3)

### Motivation

In every ONNX enhancer, `process_frame()` and `process_frame_v2()` are functionally
identical — both iterate faces and call `enhance_face()`. `process_frame_v2` just lacks
the unused `source_face` parameter.

### Changes

Handled by Phase 3's factory — `process_frame_v2` is implemented as:
```python
def process_frame_v2(temp_frame, faces=None, live_mode=False):
    return process_frame(None, temp_frame, faces=faces, live_mode=live_mode)
```

Defined once in the factory, not three times.

---

## Phase 7: Extract Single-Slot Holder Threading Pattern

**Priority**: MEDIUM | **Risk**: Medium | **Lines saved**: ~20

### Motivation

In `ui_webcam.py`, `_swap_thread_func` and `_enhancement_thread_func` follow the exact
same control flow: read input holder under lock, check seq, process, write output.

### Changes

**Create** `modules/single_slot_worker.py`:
```python
"""Reusable single-slot holder worker loop for async thread pipelines."""

def single_slot_worker_loop(
    input_holder: list,
    output_holder: list,
    lock: threading.Lock,
    stop_event: threading.Event,
    process_fn: Callable[[dict], dict],
    idle_sleep: float = 0.005,
) -> None:
    last_processed_seq = -1
    while not stop_event.is_set():
        with lock:
            inp = input_holder[0]
        if inp is None:
            time.sleep(idle_sleep)
            continue
        seq = inp['seq']
        if seq == last_processed_seq:
            time.sleep(idle_sleep)
            continue
        result = process_fn(inp)
        last_processed_seq = seq
        with lock:
            output_holder[0] = result
```

**Modify** `ui_webcam.py` — replace `_swap_thread_func` and `_enhancement_thread_func`:
```python
def _swap_process_fn(inp):
    # ... swap logic (the "do work" body) ...
    return {'frame': frame, 'seq': inp['seq']}

def _swap_thread_func(swap_input, swap_output, swap_lock, stop_event):
    single_slot_worker_loop(swap_input, swap_output, swap_lock, stop_event, _swap_process_fn)
```

### Tests

Create `tests/test_single_slot_worker.py`:
- Loop processes new inputs when seq increments
- Loop skips duplicate seq values
- Loop stops when `stop_event` is set
- `process_fn` receives correct input dict
- Thread-safety with concurrent reads/writes

---

## Phase 8: Consolidate Camera Enumeration

**Priority**: LOW | **Risk**: Medium | **Lines saved**: ~50

### Motivation

Camera enumeration exists in both `modules/ui.py` (`get_available_cameras()`) and
`modules/video_capture.py` (`VideoCapturer.__init__`).

### Changes

**Create** `modules/camera.py`:
```python
"""Platform-aware camera enumeration."""
from modules.platform_info import IS_WINDOWS, IS_APPLE_SILICON

def get_available_cameras():
    """Returns (indices, names) of available cameras.

    macOS: returns fixed [(0, 1), ("Camera 0", "Camera 1")] — no probing.
    Windows: uses pygrabber FilterGraph.
    Linux: bounded cv2.VideoCapture probe, breaks after 3 consecutive failures.
    """
    # ... consolidated from ui.py:1099-1148
```

**Modify** `ui.py` and `video_capture.py` to import from `camera.py`.

### Tests

Mock platform detection and verify correct branch for each platform. Test Linux
bounded-loop break-after-3-failures logic.

---

## Phase 9: Remove Duplicate UI Mapper Popup

**Priority**: LOW | **Risk**: Low | **Lines saved**: ~60

### Motivation

`create_source_target_popup_for_webcam()` may exist as a duplicate definition in
`ui.py` alongside the canonical version imported from `ui_mapper.py`.

### Changes

1. Verify `ui.py` imports the function from `ui_mapper.py`
2. Remove any duplicate local definition in `ui.py`
3. Verify all callers use the imported version

### Tests

Run full UI test suite. Manual test: open mapper popup from both image and webcam paths.

---

## Summary Table

| Phase | Description | Priority | Lines Saved | Risk | New Files |
|-------|-------------|----------|-------------|------|-----------|
| 1 | Platform constants | HIGH | ~4 | Very Low | `modules/platform_info.py` |
| 2 | CoreML config | HIGH | ~30 | Low | `modules/onnx_providers.py` |
| 3 | ONNX enhancer factory | CRITICAL | ~290 | Low | `modules/processors/frame/_onnx_enhancer_factory.py` |
| 4 | Pre-check helper | MEDIUM | ~40 | Low | — (added to `utilities.py`) |
| 5 | Face detection helper | LOW | ~12 | Low | — (added to `face_analyser.py`) |
| 6 | process_frame_v2 alias | MEDIUM | (in P3) | Very Low | — |
| 7 | Threading pattern | MEDIUM | ~20 | Medium | `modules/single_slot_worker.py` |
| 8 | Camera enumeration | LOW | ~50 | Medium | `modules/camera.py` |
| 9 | UI mapper popup | LOW | ~60 | Low | — |
| **Total** | | | **~500+** | | |

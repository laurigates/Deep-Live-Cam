# PRP: Apple Neural Engine Routing via CoreML Compute Units

## Objective

Configure all ONNX Runtime sessions on Apple Silicon to use `MLComputeUnits=ALL`,
enabling CoreML to route compatible operations to the Apple Neural Engine (ANE) for
improved power efficiency during live webcam sessions. Add a CLI flag for user override.

## Implementation Steps

### 1. Add `coreml_compute_units` to globals

**File:** `modules/globals.py`

Add:
```python
# CoreML Configuration (Apple Silicon only)
coreml_compute_units: str = "ALL"   # MLComputeUnits: 'ALL' (ANE+GPU+CPU), 'CPUAndGPU', 'CPUOnly'
```

### 2. Add field to `ProcessingConfig`

**File:** `modules/processing_config.py`

Add to execution configuration section:
```python
coreml_compute_units: str = "ALL"
"""CoreML compute units for Apple Silicon: 'ALL' (ANE+GPU+CPU), 'CPUAndGPU', 'CPUOnly'"""
```

### 3. Map global to config in factory

**File:** `modules/processing_config_factory.py`

In `build_config_from_globals()`:
```python
coreml_compute_units=modules.globals.coreml_compute_units,
```

In `build_config_from_cli_args()`:
```python
coreml_compute_units=getattr(args, 'coreml_compute_units', 'ALL'),
```

### 4. Update `build_providers_config` to accept parameter

**File:** `modules/onnx_providers.py`

Add `coreml_compute_units: Optional[str] = None` parameter. When `None`, read from
`modules.globals.coreml_compute_units`. Validate against `_VALID_COMPUTE_UNITS` and
fall back to `"ALL"` for invalid values.

### 5. Add CLI flag

**File:** `modules/core.py`

```python
program.add_argument(
    '--coreml-compute-units',
    help='CoreML compute units for Apple Silicon ANE routing (ALL=ANE+GPU+CPU, CPUAndGPU=GPU+CPU, CPUOnly=CPU)',
    dest='coreml_compute_units',
    default='ALL',
    choices=['ALL', 'CPUAndGPU', 'CPUOnly'],
)
```

And in `parse_args()`:
```python
modules.globals.coreml_compute_units = args.coreml_compute_units
```

### 6. Fix `face_enhancer.py` to use `build_providers_config`

**File:** `modules/processors/frame/face_enhancer.py`

Add import:
```python
from modules.onnx_providers import build_providers_config
```

In `get_face_enhancer()`, before creating the session:
```python
providers_config = build_providers_config(providers)
FACE_ENHANCER = onnxruntime.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=providers_config,
)
```

## Success Criteria

- [x] `MLComputeUnits=ALL` set by default on Apple Silicon for all ONNX sessions
- [x] `ModelCacheDirectory` configured to avoid recompilation
- [x] CLI flag `--coreml-compute-units` with choices `ALL`, `CPUAndGPU`, `CPUOnly`
- [x] `face_enhancer.py` uses `build_providers_config` (was missing, now fixed)
- [x] No effect on non-Apple platforms (`IS_APPLE_SILICON` guard unchanged)
- [x] `build_providers_config` injectable (accepts parameter for testing)
- [x] Tests cover parameter injection and globals fallback

## Testing Strategy

### Unit tests (`tests/test_onnx_providers.py`)

- [x] Test explicit `coreml_compute_units="CPUAndGPU"` override
- [x] Test explicit `coreml_compute_units="CPUOnly"` override
- [x] Test invalid value falls back to `"ALL"`
- [x] Test reads from `modules.globals.coreml_compute_units` when no parameter given
- [x] Existing tests continue to pass (no regressions)

### Manual testing (Apple Silicon hardware)

- [ ] Run with `--coreml-compute-units ALL` (default): verify ANE usage in Activity Monitor
- [ ] Run with `--coreml-compute-units CPUAndGPU`: verify no ANE traffic
- [ ] Run with `--coreml-compute-units CPUOnly`: verify CPU-only processing
- [ ] Verify no visual quality regression between compute unit modes
- [ ] Verify model cache populates in `~/.cache/deep-live-cam/coreml/` on second run

## Notes

- Face analyser (InsightFace) already filters out `CoreMLExecutionProvider` due to dynamic
  shape incompatibility — no changes needed in `face_analyser.py`
- `_onnx_enhancer.py` (GPEN models) already uses `build_providers_config` — no changes needed
- The cache directory is `~/.cache/deep-live-cam/coreml/` (XDG convention), not `models/`

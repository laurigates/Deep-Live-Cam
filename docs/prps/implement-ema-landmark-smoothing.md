# PRP: Implement EMA Landmark Smoothing

## Objective

Implement per-face exponential moving average (EMA) smoothing of bounding boxes
and keypoint landmarks in the live webcam detection pipeline to eliminate jitter
(issue #67).

## Implementation Steps

### 1. Add global configuration variables (`modules/globals.py`)

```python
landmark_smoothing: bool = False             # toggle (default: off)
landmark_smoothing_alpha: float = 0.7        # EMA weight for current frame
```

### 2. Add `ProcessingConfig` fields (`modules/processing_config.py`)

```python
landmark_smoothing: bool = False
landmark_smoothing_alpha: float = 0.7
```

### 3. Wire globals → config (`modules/processing_config_factory.py`)

Add the two new fields to `build_config_from_globals()`.

### 4. Add CLI flags (`modules/core.py`)

```
--landmark-smoothing           store_true, default=False
--landmark-smoothing-alpha     float, default=0.7, range [0.0, 1.0]
```

Assign parsed values to the corresponding globals in `parse_args()`.

### 5. Implement `LandmarkSmoother` (`modules/face_analyser.py`)

Class with:

- `alpha: float` — settable property (clamped to [0, 1])
- `IDENTITY_THRESHOLD: float = 0.7` — cosine similarity cutoff for identity match
- `smooth(faces: list) -> list` — apply in-place EMA, return same list
- `reset() -> None` — clear per-face state

Identity matching via `normed_embedding` cosine similarity (dot product since
InsightFace embeddings are already L2-normalised).

### 6. Wire into detection thread (`modules/ui_webcam.py`)

1. Import `LandmarkSmoother` from `modules.face_analyser`.
2. Extend `_detection_thread_func` signature: add optional `smoother` parameter.
3. After `detect_faces_for_webcam(...)`, apply smoothing when enabled:
   ```python
   if smoother is not None and modules.globals.landmark_smoothing:
       smoother.alpha = modules.globals.landmark_smoothing_alpha
       smoother.smooth(faces)
   elif smoother is not None:
       smoother.reset()   # clear stale state when feature toggled off
   ```
4. In `create_webcam_preview`, create `LandmarkSmoother` and pass it to the
   detection thread.

### 7. Add UI toggle (`modules/ui.py`)

Add to the `"Live Mode"` section of `_get_toggle_groups()`:

```python
(_("Landmark Smoothing"), "landmark_smoothing", False,
 _("Apply EMA smoothing to face bbox/keypoints to reduce jitter in live mode")),
```

Persist via `save_switch_states` / `load_switch_states`.

### 8. Write unit tests (`tests/test_ema_smoothing.py`)

Test categories:
- Initialisation and alpha clamping
- First-frame passthrough (no blending)
- EMA blending math (alpha=0, 0.7, 1.0)
- KPS blending
- Identity mismatch → no blending
- Face with no embedding → no blending
- Null bbox/kps handling
- Multiple faces, independent state per identity
- Face count changes (face lost, new face appears)
- `reset()` clears state and next smooth is fresh

## Success Criteria

- [x] All unit tests in `test_ema_smoothing.py` pass
- [x] `--landmark-smoothing` flag accepted by the CLI without error
- [x] `LandmarkSmoother` imported without error in headless mode
- [x] `smoother.smooth([])` clears state without crashing
- [x] No change in FPS baseline (verified manually)

## Testing Strategy

1. **Unit tests** (automated): `pytest tests/test_ema_smoothing.py`
2. **Manual smoke test** (live mode): enable smoothing, verify jitter reduction
3. **Regression**: disable smoothing, verify behaviour matches pre-feature baseline
4. **CLI test**: `python run.py --landmark-smoothing --landmark-smoothing-alpha 0.8 --help`
   (ensure args parse without error)

## Notes

- `LandmarkSmoother` modifies `face.bbox` and `face.kps` in-place.  InsightFace
  `Face` objects are dict subclasses and support attribute assignment safely.
- The smoother lives in the detection thread — it is never accessed from the
  processing or swap thread, so no additional locking is needed.
- The `reset()` call when smoothing is disabled ensures that toggling the feature
  on/off mid-session does not produce blending artefacts.

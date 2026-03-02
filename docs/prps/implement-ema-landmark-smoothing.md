# PRP: Implement EMA Landmark Smoothing

## Objective

Reduce visible jitter in live webcam face-swap mode by applying Exponential Moving Average
(EMA) smoothing to face bounding boxes and landmark keypoints.

## Implementation Steps

1. **Add globals** (`modules/globals.py`):
   - `landmark_smoothing: bool = False`
   - `landmark_smoothing_alpha: float = 0.7`

2. **Add ProcessingConfig fields** (`modules/processing_config.py`):
   - `landmark_smoothing: bool = False`
   - `landmark_smoothing_alpha: float = 0.7`

3. **Update factory** (`modules/processing_config_factory.py`):
   - Map globals → config in `build_config_from_globals()`

4. **Add CLI flags** (`modules/core.py`):
   - `--landmark-smoothing` (store_true)
   - `--landmark-smoothing-alpha` (float, default=0.7)

5. **Implement `LandmarkSmoother`** (`modules/face_analyser.py`):
   - Per-identity state keyed by embedding cosine similarity (threshold 0.7)
   - EMA: `smoothed = alpha * current + (1 - alpha) * prev`
   - Modifies `.bbox` and `.kps` attributes in-place on InsightFace Face objects
   - `reset()` clears all state

6. **Wire into detection thread** (`modules/ui_webcam.py`):
   - Create `LandmarkSmoother` instance per webcam session
   - Pass to `_detection_thread_func` as `smoother` parameter
   - Apply after `detect_faces_for_webcam()` when `landmark_smoothing` is enabled
   - Clear smoother on session cleanup

7. **Add UI toggle** (`modules/ui.py`):
   - Add "Landmark Smoothing" switch to Live Mode tab via `_get_switch_defs()`
   - Persist in `save_switch_states()` / `load_switch_states()`

8. **Tests** (`tests/test_ema_smoothing.py`):
   - EMA math correctness (frame 1, frame 2, alpha=1.0)
   - Identity matching (same/different embedding)
   - Many-faces mode
   - Reset and empty detection
   - Convergence

9. **Documentation**:
   - ADR 0013 (`docs/adrs/0013-ema-landmark-smoothing.md`)
   - PRD (`docs/prds/ema-landmark-smoothing.md`)
   - This PRP

## Success Criteria

- [x] Unit tests pass for all EMA logic
- [x] No changes to batch/video processing code paths
- [x] Feature off by default — toggle via CLI or UI
- [x] Alpha guide documented for users

## Testing Strategy

- Unit: mock InsightFace `Face` objects, verify EMA formula applied to `.bbox` and `.kps`
- Integration: run `pytest tests/test_ema_smoothing.py` in CI
- Manual: enable `--landmark-smoothing`, compare live preview with/without for jitter

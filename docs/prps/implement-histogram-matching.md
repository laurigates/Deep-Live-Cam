# PRP: Implement Histogram Matching Color Correction

## Objective

Add per-channel CDF histogram matching in LAB color space as a selectable color
correction mode for face swap output, alongside the existing LAB mean/std transfer.

## Implementation Steps

1. **`modules/processors/frame/face_masking.py`**
   - Add `apply_histogram_matching(source, target) -> np.ndarray`
   - Algorithm: convert both images to LAB; per channel, compute `np.unique`-based
     CDFs for source and target; use `np.interp` to map source values to target values
     via CDF; convert result back to BGR.
   - Guard clauses: return source unchanged if source or target is None or empty.

2. **`modules/globals.py`**
   - Add `color_correction_mode: str = 'none'` (values: `'none'`, `'lab'`, `'histogram'`)
   - Keep existing `color_correction: bool` (unrelated BGR/RGB webcam fix)

3. **`modules/processing_config.py`**
   - Add `color_correction_mode: str = 'none'` field with docstring

4. **`modules/processing_config_factory.py`**
   - Map `color_correction_mode` from globals in `build_config_from_globals()`

5. **`modules/processors/frame/face_swapper.py`**
   - Import `apply_histogram_matching` from face_masking
   - Replace `swap_color_transfer` checks with `color_correction_mode` dispatch:
     - `'lab'` → `apply_color_transfer()`
     - `'histogram'` → `apply_histogram_matching()`
     - Backward compat: if mode is `'none'` but `swap_color_transfer` is True, use `'lab'`
   - Apply to both `batch_swap_faces` and `swap_face` paths

6. **`modules/core.py`**
   - Add `--color-correction` argument with `choices=['none', 'lab', 'histogram']`
   - Wire to `modules.globals.color_correction_mode`

7. **`modules/ui.py`**
   - Add `_add_color_correction_to_processing_tab()` function
   - Call it from `_add_settings_tabview()` for the Processing tab
   - Persist `color_correction_mode` in `save_switch_states()` / `load_switch_states()`

8. **`tests/test_color_correction.py`**
   - Guard clause tests (None/empty inputs)
   - Output property tests (shape, dtype, range, non-mutation)
   - Histogram distance test (result closer to target than source)
   - Identity test (matching image to itself ≈ identity)
   - ProcessingConfig field tests
   - Integration: both functions modify crop; produce different outputs

## Success Criteria

- [ ] All tests in `test_color_correction.py` pass
- [ ] Existing tests in `test_preprocessing_phase3.py` still pass (backward compat)
- [ ] `uv run run.py --color-correction histogram -s src.jpg -t tgt.mp4 -o out.mp4` works
- [ ] Processing tab in UI shows the Color Correction dropdown

## Testing Strategy

Run the test suite:
```bash
uv run pytest tests/test_color_correction.py -v
uv run pytest tests/test_preprocessing_phase3.py -v
```

Visual verification:
- Run with `--color-correction lab` on a cross-skin-tone source/target pair
- Run with `--color-correction histogram` on the same pair
- Confirm histogram mode produces noticeably better blending
- Confirm `--color-correction none` produces no correction (default)

# Feature: Color Correction Modes for Face Swap

## Overview

Add a selectable color correction mode that improves skin-tone blending between
the source and target faces after swapping.  The existing LAB mean/std transfer
handles moderate differences; histogram matching provides stronger correction for
cross-skin-tone swaps.

## User Stories

- As a user swapping faces between people of different ethnicities, I want stronger
  color correction so the swapped face blends naturally into the target skin tone.
- As a user swapping faces with similar skin tones, I want to be able to disable
  or use a lighter correction mode to avoid over-processing.
- As a CLI user, I want a `--color-correction` flag so I can automate batch
  processing with the correct correction level.

## Acceptance Criteria

- [ ] `--color-correction [none|lab|histogram]` CLI flag is available
- [ ] "Color Correction" dropdown is present in the Processing tab of the UI
- [ ] `'none'` disables all correction (default, no behaviour change)
- [ ] `'lab'` applies the existing LAB mean/std transfer
- [ ] `'histogram'` applies per-channel CDF histogram matching in LAB space
- [ ] Visibly better skin-tone blending on cross-ethnicity swaps when using `'histogram'`
- [ ] No color artifacts on same-skin-tone swaps when using `'histogram'`
- [ ] Both single-face and multi-face modes work correctly with all correction modes
- [ ] Unit tests cover histogram matching output properties and histogram distance
- [ ] No measurable FPS regression in live mode (< 2 ms per face added)
- [ ] Selected mode is persisted across sessions via the state file

## Non-Functional Requirements

- **Performance**: Histogram matching ≤ 2 ms per face on CPU (128 × 128 crop)
- **Compatibility**: Existing `swap_color_transfer` bool flag continues to work
- **Platform**: Works on macOS ARM, Linux CUDA, and CPU-only environments

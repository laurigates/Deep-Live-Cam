# ADR-0012: Color Correction Modes for Face Swap

## Status
Accepted

## Context

After a face is swapped, the swapped crop often has a different colour distribution
from the target face region, producing a visible skin-tone mismatch.  The project
already contains a LAB mean/std transfer (`apply_color_transfer`) that reduces
moderate differences.  However, when the source and target have very different
skin tones (e.g. cross-ethnicity swaps, extreme lighting), a stronger correction
is needed.

Three candidate approaches were considered:

| Approach | Mechanism | Strength | Risk |
|----------|-----------|----------|------|
| **None** | No correction | — | Mismatch visible on different tones |
| **LAB mean/std** | Shift mean & scale std per channel | Moderate | Can over-flatten low-contrast faces |
| **Histogram matching** | Per-channel CDF interpolation | Strong | May introduce mild artefacts on identical tones |
| cv2.createCLAHE | Adaptive histogram equalisation on L only | Luminance-only | Ignores colour channels |

## Decision

Introduce a `color_correction_mode` string setting with three values: `'none'`,
`'lab'`, and `'histogram'`.  The existing `swap_color_transfer` bool is preserved
for backward compatibility; when `color_correction_mode` is `'none'` and
`swap_color_transfer` is `True`, the system falls back to LAB mode.

Histogram matching uses per-channel CDF interpolation in LAB colour space, applied
to the swapped face crop before paste-back — the same stage as the existing LAB
transfer.  The implementation is ~30 lines of pure NumPy/OpenCV with no additional
dependencies.

CLAHE was not selected as a first-class mode because it only addresses luminance,
not the colour (a/b) channels that carry skin-tone information.  It can be added
later as a fourth option without changing the API.

## Consequences

**Positive**
- Users can choose the correction strength that suits their workflow
- No FPS regression — histogram matching adds ≈1 ms per face (dominated by `np.unique`)
- LAB transfer preserved verbatim; no behaviour change for existing users
- CLI (`--color-correction`) and UI dropdown surface the new option consistently

**Negative**
- One additional string global (`color_correction_mode`) added to `modules.globals`
- Histogram matching may over-correct when source and target have the same skin tone;
  mitigated by the `'none'` default and user-selectable modes

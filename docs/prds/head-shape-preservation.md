# Feature: Head Shape Preservation During Face Swap

## Overview

Currently, face swapping replaces the face texture within the **target person's head boundary**. The swapped face inherits the target's jawline, forehead, and overall head shape. This feature would preserve the **source face's head shape** (jawline, forehead contour, skull shape) when pasting the swap result, producing a more convincing result where the swapped person's actual head proportions are maintained.

## Problem Statement

The face swap pipeline uses InsightFace's 5-point alignment (two eyes, nose, two mouth corners) to compute a similarity transform (scale + rotation + translation). The aligned 128px crop is fed through the swap model, and the result is warped back using the inverse of the same affine matrix. The paste-back boundary is derived from the **target face's 2D landmarks** (106-point outline), meaning:

- The jawline contour comes from the target face
- The forehead/hairline boundary comes from the target face
- The seamless blend (Poisson) and occlusion mask both operate within this target-derived boundary

When the source and target have significantly different face shapes (e.g., round vs. angular jaw, wide vs. narrow forehead), the swapped face looks unnatural because the face texture doesn't match the head boundary it's placed within.

## User Stories

- As a user performing a face swap between two people with different face shapes, I want the output to preserve the source person's jawline and head proportions so the result looks natural
- As a user in live mode, I want a toggle to enable/disable head shape preservation so I can compare results quickly
- As a user, I want the feature to work with both single-face and multi-face modes

## Technical Approach

### Option A: Boundary Warping (Recommended)

After computing the swapped face crop, warp the **paste-back boundary** to match the source face's landmark contour rather than the target's:

1. Extract source face 106-point landmarks (jawline indices 0-32, forehead estimated from brow indices 33-43)
2. Extract target face 106-point landmarks (same indices)
3. Compute a thin-plate spline (TPS) or piecewise affine warp that maps target boundary points to source boundary points
4. Apply this boundary warp to the face mask used in `_paste_back()` and `_apply_poisson_blend()`
5. Optionally warp the surrounding skin region to smoothly transition between the source-shaped face and the target's neck/hair

**Pros**: Preserves neural network output quality; only modifies the boundary
**Cons**: May produce artifacts at the neck/hair boundary; TPS is moderately expensive

### Option B: Full Face Shape Transfer

Apply a dense warp to the entire face region to deform the target's face shape toward the source's proportions:

1. Compute landmark correspondence between source and target 106-point sets
2. Generate a dense deformation field using Delaunay triangulation or TPS
3. Warp the entire target face region before running the swap model
4. Run swap on the pre-warped face
5. Inverse-warp the result back

**Pros**: More thorough shape transfer
**Cons**: Two extra warp operations per frame (expensive for live mode); may degrade swap model quality if the pre-warp is too aggressive

### Option C: Neural Approach (Future)

Train or fine-tune the swap model to also transfer facial geometry, not just texture. This is a research-level effort and out of scope for initial implementation.

## Acceptance Criteria

- [ ] Toggle in Processing tab: "Preserve Source Head Shape" (default off)
- [ ] When enabled, the paste-back boundary follows the source face's jawline/forehead contour
- [ ] Works in both live webcam mode and image/video processing
- [ ] Works with single-face, many-faces, and map-faces modes
- [ ] No more than 15% FPS regression in live mode on Apple Silicon M4 Pro
- [ ] Smooth boundary transition (no visible seam at jawline/neck)
- [ ] Graceful fallback to target boundary when source landmarks are unavailable

## Non-Functional Requirements

- **Performance**: Must maintain real-time capability (>15 FPS) in live mode
- **Quality**: Boundary warp should not introduce visible distortion in the neck/hair region
- **Compatibility**: Must work with all swap models (inswapper, Ghost, HyperSwap)

## Dependencies

- 106-point landmark detection (already available via `face.landmark_2d_106`)
- Existing `create_face_mask()` in `face_masking.py` (boundary source)
- OpenCV thin-plate spline or scikit-image piecewise affine warp

## Out of Scope

- Full 3D head reconstruction
- Hair transfer or hairstyle matching
- Body shape modification beyond the face/jaw region

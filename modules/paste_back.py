"""Shared paste-back primitives for face processing pipeline.

Provides composable building blocks used by the three paste-back paths:
- face_swapper: diff-mask + erode/blur blending
- face_enhancer: feathered-mask blending (cached)
- _onnx_enhancer: feathered-mask blending (inline)

Each caller retains its distinct algorithm but delegates shared sub-operations
here instead of reimplementing.
"""

import cv2
import numpy as np


def inverse_affine_warp(
    image: np.ndarray,
    affine_matrix: np.ndarray,
    target_size: tuple[int, int],
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value=0,
) -> np.ndarray:
    """Warp an image from aligned space back to frame space via inverse affine.

    Args:
        image: Aligned image (face crop or mask) in any dtype.
        affine_matrix: 2x3 forward affine matrix (aligned → frame NOT needed;
            this function computes the inverse internally).
        target_size: (width, height) of the output frame.
        border_mode: OpenCV border mode for out-of-bounds pixels.
        border_value: Fill value for BORDER_CONSTANT mode.

    Returns:
        Warped image with shape (height, width, ...) matching target_size.
    """
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    return cv2.warpAffine(
        image,
        inv_matrix,
        target_size,
        borderMode=border_mode,
        borderValue=border_value,
    )


def create_feathered_mask(
    size: int,
    border_fraction: float = 0.05,
) -> np.ndarray:
    """Create a square feathered blending mask with soft edges.

    The mask is 1.0 in the interior and ramps to 0.0 at the borders.
    Returns a 3-channel float32 mask suitable for direct multiplication
    with BGR images.

    Args:
        size: Width and height of the square mask.
        border_fraction: Fraction of size used for the feathered border.
            E.g., 0.05 means 5% of the size on each edge.

    Returns:
        Float32 array of shape (size, size, 3) with values in [0, 1].
    """
    border = max(1, int(size * border_fraction))

    mask = np.ones((size, size), dtype=np.float32)
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)

    mask[:border, :] *= ramp_up[:, None]
    mask[-border:, :] *= ramp_down[:, None]
    mask[:, :border] *= ramp_up[None, :]
    mask[:, -border:] *= ramp_down[None, :]

    return np.stack([mask] * 3, axis=-1)


def create_feathered_mask_1c(
    size: int,
    border_fraction: float = 1 / 16,
) -> np.ndarray:
    """Create a single-channel feathered blending mask.

    Like create_feathered_mask but returns a 2D (size, size) float32 array.
    Uses multiplicative min-ramp at corners (same algorithm as _onnx_enhancer).

    Args:
        size: Width and height of the square mask.
        border_fraction: Fraction of size used for the feathered border.

    Returns:
        Float32 array of shape (size, size) with values in [0, 1].
    """
    border = max(1, int(size * border_fraction))

    mask = np.ones((size, size), dtype=np.float32)
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)

    mask[:border, :] = ramp_up[:, np.newaxis]
    mask[-border:, :] = ramp_down[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], ramp_up[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], ramp_down[np.newaxis, :])

    return mask


def blend_with_mask(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Alpha-blend foreground onto background using a float mask.

    Args:
        foreground: Warped face image (float32 or uint8).
        background: Original frame (float32 or uint8).
        mask: Blending weights in [0, 1]. Can be 2D (H, W) or 3D (H, W, C).
            1.0 = fully foreground, 0.0 = fully background.

    Returns:
        Blended result as uint8.
    """
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    if mask.ndim == 2:
        m = mask[:, :, np.newaxis]
    else:
        m = mask

    result = fg * m + bg * (1.0 - m)
    return np.clip(result, 0, 255).astype(np.uint8)

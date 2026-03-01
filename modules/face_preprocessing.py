"""Shared preprocessing/postprocessing for face enhancement ONNX models.

Both GFPGAN (face_enhancer.py) and GPEN-BFR (_onnx_enhancer.py) use
identical BGR→RGB→[-1,1]→NCHW transforms. This module provides the
canonical implementation so callers don't duplicate the logic.

Swapper preprocessing is intentionally separate — it uses model-specific
cv2.dnn.blobFromImage with per-channel mean/std.
"""

import cv2
import numpy as np


def preprocess_enhancement_input(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 face crop → NCHW float32 [-1, 1] for enhancement models.

    Args:
        face_bgr: Aligned face crop in BGR uint8 format (H, W, 3).

    Returns:
        Input tensor of shape (1, 3, H, W) with values in [-1, 1].
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 255.0 * 2.0 - 1.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def postprocess_enhancement_output(tensor: np.ndarray) -> np.ndarray:
    """NCHW float32 [-1, 1] → BGR uint8 face image.

    Args:
        tensor: Model output tensor of shape (1, 3, H, W) or (3, H, W).

    Returns:
        BGR uint8 image of shape (H, W, 3).
    """
    face = np.squeeze(tensor)  # remove batch dim → (3, H, W)
    face = np.transpose(face, (1, 2, 0))  # CHW → HWC
    face = (face + 1.0) / 2.0
    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

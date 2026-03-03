"""XSeg-based face occlusion masking.

Detects occluding objects (hands, glasses, microphones, hair) on face-aligned
crops and returns a per-pixel mask that preserves target pixels in occluded
regions during face swap.
"""
import threading
import numpy as np
import onnxruntime
import os

import modules.globals
from modules.status_bus import BUS
from modules.paths import MODELS_DIR
from modules.onnx_providers import build_providers_config
from modules.utilities import conditional_download

FACE_OCCLUDER: onnxruntime.InferenceSession | None = None
_OCCLUDER_LOCK = threading.Lock()
NAME = "DLC.FACE-OCCLUDER"

_XSEG_MODEL = {
    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_2.onnx',
    'file': 'xseg_2.onnx',
    # SHA-256 sourced from huggingface.co/facefusion/models-3.1.0
    # Verify with: sha256sum models/xseg_2.onnx
    'sha256': 'cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042',
}


def pre_check() -> bool:
    """Download XSeg model if missing."""
    model_path = os.path.join(MODELS_DIR, _XSEG_MODEL['file'])
    if not os.path.exists(model_path):
        BUS.publish(f"Downloading {_XSEG_MODEL['file']}...", NAME)
    conditional_download(
        MODELS_DIR,
        [_XSEG_MODEL['url']],
        expected_checksums={_XSEG_MODEL['file']: _XSEG_MODEL['sha256']},
    )
    if not os.path.exists(model_path):
        BUS.publish(f"XSeg model not found at {model_path}. Download may have failed.", NAME)
        return False
    return True


def get_face_occluder(providers: list | None = None) -> onnxruntime.InferenceSession | None:
    """Lazy singleton loader for XSeg occlusion model."""
    global FACE_OCCLUDER
    if FACE_OCCLUDER is None:
        with _OCCLUDER_LOCK:
            if FACE_OCCLUDER is None:
                _providers = providers if providers is not None else modules.globals.execution_providers
                providers_config = build_providers_config(_providers)
                model_path = os.path.join(MODELS_DIR, _XSEG_MODEL['file'])
                if not os.path.exists(model_path):
                    BUS.publish("XSeg model not found — downloading now...", NAME)
                    if not pre_check():
                        return None
                try:
                    FACE_OCCLUDER = onnxruntime.InferenceSession(
                        model_path, providers=providers_config,
                    )
                    BUS.publish("XSeg occlusion model loaded.", NAME)
                except Exception as e:
                    BUS.publish(f"Error loading XSeg model: {e}", NAME)
                    FACE_OCCLUDER = None
    return FACE_OCCLUDER


def create_occlusion_mask(crop_frame: np.ndarray) -> np.ndarray:
    """Run XSeg on a face-aligned crop.

    Returns float32 mask with same H x W as input.
    1.0 = unoccluded face region (swap), 0.0 = occluded region (preserve target).
    """
    import cv2

    session = get_face_occluder()
    if session is None:
        return np.ones(crop_frame.shape[:2], dtype=np.float32)

    orig_h, orig_w = crop_frame.shape[:2]

    # Preprocess: resize to 256x256, float32, normalize to [0, 1], NHWC
    resized = cv2.resize(crop_frame, (256, 256))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = input_tensor[np.newaxis]  # (1, 256, 256, 3)

    # ONNX inference
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    output = session.run([out_name], {inp_name: input_tensor})[0]

    # Postprocess: squeeze, clip to [0, 1], resize back
    mask = output.squeeze()
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Take first channel if multi-channel
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    # Resize back to original crop dimensions
    if mask.shape != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h))

    # Smooth edges with Gaussian blur
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask

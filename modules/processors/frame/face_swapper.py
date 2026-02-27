from typing import Any, List, Optional, Tuple
import cv2
import insightface
import logging
import threading
import numpy as np
import modules.globals
import modules.processors.frame.core
from modules.face_map_store import STORE as _MAP_STORE
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
from modules.gpu_processing import gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted, gpu_resize, gpu_cvt_color
from modules.paths import MODELS_DIR
from modules.platform_info import IS_APPLE_SILICON
from modules.onnx_providers import build_providers_config
from modules.processors.frame.face_masking import (
    apply_color_transfer,
    create_face_mask,
    create_lower_mouth_mask,
    draw_mouth_mask_visualization,
    apply_mouth_area,
)
import os
import time

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
# Per-thread previous frame for interpolation (avoids cross-thread contamination)
_THREAD_LOCAL = threading.local()


def _get_previous_frame():
    return getattr(_THREAD_LOCAL, 'previous_frame_result', None)


def _set_previous_frame(frame):
    _THREAD_LOCAL.previous_frame_result = frame
# --- END: Added for Interpolation ---

FACE_DETECTION_CACHE = {}  # Cache face detections (2 fixed keys: 'faces', 'timestamp')
LAST_DETECTION_TIME = 0

models_dir = MODELS_DIR

def pre_check() -> bool:
    # Use models_dir instead of abs_dir to save to the correct location
    download_directory_path = models_dir

    # Make sure the models directory exists, catch permission errors if they occur
    try:
        os.makedirs(download_directory_path, exist_ok=True)
    except OSError as e:
        print(f"{NAME}: Failed to create directory {download_directory_path}: {e}")
        return False

    model_file = "inswapper_128_fp16.onnx"
    model_path = os.path.join(download_directory_path, model_file)
    if not os.path.exists(model_path):
        update_status(f"Downloading {model_file}...", NAME)
    # Use the direct download URL from Hugging Face
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
        ],
    )
    if not os.path.exists(model_path):
        update_status(f"Model not found at {model_path}. Download may have failed.", NAME)
        return False
    return True


def pre_start() -> bool:
    # Simplified pre_start, assuming checks happen before calling process functions
    model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    if not os.path.exists(model_path):
        update_status(f"Model not found: {model_path}. Please download it.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    # Add other essential checks if needed, e.g., target/source path validity
    return True


def get_face_swapper(providers: list | None = None) -> Any:
    """Return the face swapper singleton, initialising it if needed.

    *providers* overrides ``modules.globals.execution_providers`` when given,
    enabling tests and callers to inject providers without touching globals.
    """
    global FACE_SWAPPER

    if FACE_SWAPPER is None:
        with THREAD_LOCK:
            if FACE_SWAPPER is None:
                model_name = "inswapper_128_fp16.onnx"
                model_path = os.path.join(models_dir, model_name)
                update_status(f"Loading face swapper model from: {model_path}", NAME)
                try:
                    _providers = providers if providers is not None else modules.globals.execution_providers
                    providers_config = build_providers_config(_providers)

                    FACE_SWAPPER = insightface.model_zoo.get_model(
                        model_path,
                        providers=providers_config,
                    )
                    update_status("Face swapper model loaded successfully.", NAME)

                    # Prefer MLX over ONNX Runtime on Apple Silicon — runs the full graph
                    # natively on the Metal GPU (8–9× faster than CoreML EP in benchmarks).
                    # Falls through to CoreML .mlpackage, then ONNX Runtime CoreML EP.
                    mlx_session_loaded = False
                    if IS_APPLE_SILICON:
                        try:
                            from modules.mlx_inswapper import MLXSessionWrapper
                            mlx_session = MLXSessionWrapper.load(model_path)
                            if mlx_session is not None:
                                FACE_SWAPPER.session = mlx_session
                                update_status("Using MLX inference (native Metal GPU).", NAME)
                                mlx_session_loaded = True
                        except Exception as mlx_err:
                            update_status(f"MLX session load failed, trying CoreML: {mlx_err}", NAME)

                    # Fallback: direct CoreML model (.mlpackage) over ONNX Runtime CoreML EP.
                    # Generate with: uv run scripts/convert_to_coreml.py
                    mlpackage_path = os.path.join(models_dir, "inswapper_128.mlpackage")
                    if IS_APPLE_SILICON and not mlx_session_loaded and os.path.exists(mlpackage_path):
                        try:
                            from modules.coreml_session import CoreMLSessionWrapper
                            coreml_session = CoreMLSessionWrapper.load(mlpackage_path)
                            if coreml_session is not None:
                                FACE_SWAPPER.session = coreml_session
                                update_status("Using direct CoreML model (bypassing ONNX Runtime).", NAME)
                        except Exception as cml_err:
                            update_status(f"CoreML session load failed, using ONNX Runtime: {cml_err}", NAME)

                    # Warmup inference: trigger JIT compilation / compute plan caching
                    # so the first real inference call has no latency spike.
                    if mlx_session_loaded or any(
                        (p[0] if isinstance(p, tuple) else p) == "CoreMLExecutionProvider"
                        for p in providers_config
                    ) or (IS_APPLE_SILICON and os.path.exists(mlpackage_path)):
                        try:
                            session = FACE_SWAPPER.session
                            input_feed = {
                                inp.name: np.zeros(
                                    [d if isinstance(d, int) and d > 0 else 1
                                     for d in inp.shape],
                                    dtype=np.float32,
                                )
                                for inp in session.get_inputs()
                            }
                            session.run(None, input_feed)
                            update_status("Warmup inference complete.", NAME)
                        except Exception as warmup_err:
                            update_status(
                                f"Warmup skipped (non-fatal): {warmup_err}", NAME
                            )
                except Exception as e:
                    update_status(f"Error loading face swapper model: {e}", NAME)
                    FACE_SWAPPER = None
                    return None
    return FACE_SWAPPER


def _apply_mouth_mask(swapped: Frame, target_face: Face, original: Frame) -> Frame:
    if not getattr(modules.globals, "mouth_mask", False):
        return swapped

    face_mask = create_face_mask(target_face, original)
    mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
        create_lower_mouth_mask(target_face, original)
    )

    if mouth_cutout is not None and mouth_box != (0, 0, 0, 0):
        swapped = apply_mouth_area(
            swapped, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if getattr(modules.globals, "show_mouth_mask_box", False):
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped = draw_mouth_mask_visualization(
                swapped, target_face, mouth_mask_data
            )

    return swapped


def _apply_poisson_blend(swapped: Frame, target_face: Face, original: Frame, pre_swap: Frame) -> Frame:
    if not getattr(modules.globals, "poisson_blend", False):
        return swapped

    face_mask = create_face_mask(target_face, original)
    if face_mask is None:
        return swapped

    y_indices, x_indices = np.where(face_mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return swapped

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

    src_crop = swapped[y_min : y_max + 1, x_min : x_max + 1]
    mask_crop = face_mask[y_min : y_max + 1, x_min : x_max + 1]

    try:
        swapped = cv2.seamlessClone(
            src_crop,
            pre_swap,
            mask_crop,
            center,
            cv2.NORMAL_CLONE,
        )
    except Exception as e:
        print(f"Poisson blending failed: {e}")

    return swapped


logger = logging.getLogger(NAME)

# Track whether batch inference fallback warning has been issued
_batch_fallback_warned = False


def _clamp_opacity() -> float:
    """Read the current opacity from globals, clamped to [0.0, 1.0]."""
    return max(0.0, min(1.0, getattr(modules.globals, "opacity", 1.0)))


def _paste_back(bgr_fake: np.ndarray, aimg: np.ndarray, M: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """Paste an aligned face crop back onto the target image using inverse affine warp and blending.

    Replicates the INSwapper paste_back logic so it can be reused by both
    single-face and batch paths.
    """
    fake_diff = np.abs(bgr_fake.astype(np.float32) - aimg.astype(np.float32)).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0

    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)

    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)

    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255

    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
        return target_img
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
    img_mask /= 255
    fake_diff /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    return fake_merged.astype(np.uint8)


def _paste_scale_from_M(M: np.ndarray, max_k: float = 4.0) -> float:
    """Return the upscale factor k for pre-paste upscaling.

    M maps the full frame to a 128×128 crop.  The scale embedded in M tells us
    how many source pixels map to one 128px output pixel.  Upscaling by k before
    the inverse-affine paste ensures we paste a resolution-appropriate crop rather
    than stretching a tiny 128×128 result over a large face region.
    """
    scale_to_128 = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
    if scale_to_128 <= 0:
        return 1.0
    k = 1.0 / scale_to_128
    return float(np.clip(k, 1.0, max_k))


def _upscale_crop_for_paste(
    bgr_fake: np.ndarray,
    aimg: np.ndarray,
    M: np.ndarray,
    k: float,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> tuple:
    """Return (bgr_fake_up, aimg_up, M_scaled) upscaled by factor k.

    Both bgr_fake and aimg must be upscaled together so _paste_back can compute
    fake_diff = |bgr_fake - aimg| with matching shapes.  M is scaled by k so
    the inverse-affine warp maps the upscaled crop correctly onto the frame.
    """
    if k <= 1.0 + 1e-3:
        return bgr_fake, aimg, M
    target = (int(round(bgr_fake.shape[1] * k)), int(round(bgr_fake.shape[0] * k)))
    bgr_up = gpu_resize(bgr_fake, target, interpolation=interpolation)
    aimg_up = gpu_resize(aimg, target, interpolation=interpolation)
    M_scaled = M * k  # broadcast over 2×3; maps k×128 → full frame via invertAffine
    return bgr_up, aimg_up, M_scaled


def batch_swap_faces(
    source_faces: List[Face],
    target_faces: List[Face],
    temp_frame: Frame,
) -> Frame:
    """Swap multiple faces in a single ONNX inference call.

    Pre/post-processing is per-face (alignment and inverse warp are face-specific),
    but the expensive model inference is batched into one session.run() call.

    Falls back to sequential swap_face() calls if session.run() raises an error
    (e.g., CoreML rejecting dynamic batch).
    """
    from insightface.utils import face_align

    global _batch_fallback_warned

    face_swapper = get_face_swapper()
    if face_swapper is None:
        return temp_frame

    session = face_swapper.session
    emap = face_swapper.emap
    input_size = face_swapper.input_size
    input_mean = face_swapper.input_mean
    input_std = face_swapper.input_std

    # Collect per-face pre-processing results
    blobs = []
    latents = []
    aligned_imgs = []
    affine_matrices = []
    valid_indices = []  # indices into source_faces/target_faces for valid pairs

    for i, (source_face, target_face) in enumerate(zip(source_faces, target_faces)):
        if source_face is None or target_face is None:
            continue
        if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
            continue

        aimg, M = face_align.norm_crop2(temp_frame, target_face.kps, input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0 / input_std, input_size,
            (input_mean, input_mean, input_mean), swapRB=True
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, emap)
        latent /= np.linalg.norm(latent)

        blobs.append(blob)
        latents.append(latent)
        aligned_imgs.append(aimg)
        affine_matrices.append(M)
        valid_indices.append(i)

    if not blobs:
        return temp_frame

    # Attempt batched inference
    try:
        batch_blob = np.concatenate(blobs, axis=0).astype(np.float32)  # (N, 3, 128, 128)
        batch_latent = np.concatenate(latents, axis=0).astype(np.float32)  # (N, 512)

        preds = session.run(
            face_swapper.output_names,
            {face_swapper.input_names[0]: batch_blob, face_swapper.input_names[1]: batch_latent},
        )[0]  # (N, 3, 128, 128)
    except Exception as e:
        if not _batch_fallback_warned:
            logger.warning("Batch inference failed (%s), falling back to sequential swap", e)
            _batch_fallback_warned = True
        # Fallback: sequential swap_face calls
        result = temp_frame.copy()
        for source_face, target_face in zip(source_faces, target_faces):
            result = swap_face(source_face, target_face, result)
        return result

    # Post-processing: paste each face back onto the frame
    opacity = _clamp_opacity()
    original_frame = temp_frame if opacity >= 1.0 else temp_frame.copy()
    result = temp_frame.copy()

    for j, idx in enumerate(valid_indices):
        target_face = target_faces[idx]
        aimg = aligned_imgs[j]
        M = affine_matrices[j]

        # Denormalize prediction
        img_fake = preds[j].transpose((1, 2, 0))
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        # Optionally upscale crop before paste-back to reduce stretch artifact
        if getattr(modules.globals, "prepaste_upscale", True):
            k = _paste_scale_from_M(M)
            bgr_fake, aimg, M = _upscale_crop_for_paste(bgr_fake, aimg, M, k)

        # Paste back onto result frame
        result = _paste_back(bgr_fake, aimg, M, result)

        # Apply mouth mask and Poisson blend per-face
        result = _apply_mouth_mask(result, target_face, temp_frame)
        result = _apply_poisson_blend(result, target_face, temp_frame, original_frame)

    if opacity < 1.0:
        result = gpu_add_weighted(
            original_frame.astype(np.uint8), 1 - opacity,
            result.astype(np.uint8), opacity, 0,
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping with better memory management and performance."""
    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame

    # Safety check for faces
    if source_face is None or target_face is None:
        return temp_frame
    if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
        return temp_frame

    # Store a copy of the original frame before swapping for opacity blending
    opacity = _clamp_opacity()
    original_frame = temp_frame if opacity >= 1.0 else temp_frame.copy()

    # Pre-swap Input Check with optimization
    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    # Apply the face swap with optimized memory handling
    try:
        from insightface.utils import face_align as _face_align

        # Ensure contiguous memory layout for better performance on all platforms
        if not temp_frame.flags['C_CONTIGUOUS']:
            temp_frame = np.ascontiguousarray(temp_frame)

        # Use paste_back=False to get the raw crop + affine matrix so we can
        # optionally upscale the crop before warping it back onto the frame.
        bgr_fake, M = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=False
        )

        if bgr_fake is None or M is None:
            return original_frame

        # Retrieve the aligned crop that was used as the swap base (needed by
        # _paste_back to compute the diff mask).
        aimg, _ = _face_align.norm_crop2(
            temp_frame, target_face.kps, face_swapper.input_size[0]
        )

        # Optionally upscale crop before paste-back to reduce stretch artifact
        if getattr(modules.globals, "prepaste_upscale", True):
            k = _paste_scale_from_M(M)
            bgr_fake, aimg, M = _upscale_crop_for_paste(bgr_fake, aimg, M, k)

        swapped_frame_raw = _paste_back(bgr_fake, aimg, M, temp_frame)

        if swapped_frame_raw is None:
            return original_frame

        if not isinstance(swapped_frame_raw, np.ndarray):
            return original_frame

        if swapped_frame_raw.shape != temp_frame.shape:
            try:
                swapped_frame_raw = gpu_resize(swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0]))
            except Exception:
                return original_frame

        swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Error during face swap using face_swapper.get: {e}")
        return original_frame

    # --- Post-swap Processing (Masking, Opacity, etc.) ---
    # Now, work with the guaranteed uint8 'swapped_frame'

    swapped_frame = _apply_mouth_mask(swapped_frame, target_face, temp_frame)
    swapped_frame = _apply_poisson_blend(swapped_frame, target_face, temp_frame, original_frame)

    # Apply opacity blend between the original frame and the swapped frame
    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    # Blend the original_frame with the (potentially mouth-masked) swapped_frame
    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


# --- START: Mac M1-M5 Optimized Face Detection ---
def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    """Optimized face detection for live mode on Apple Silicon"""
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE
    
    if not use_cache:
        # Standard detection (no caching)
        if modules.globals.many_faces:
            return get_many_faces(frame)
        else:
            face = get_one_face(frame)
            return [face] if face else None

    # TTL-based detection caching — skips detection when the previous result
    # is still fresh (within DETECTION_INTERVAL).  Previously gated behind
    # IS_APPLE_SILICON; now enabled on all platforms for live mode.
    current_time = time.time()
    time_since_last = current_time - LAST_DETECTION_TIME
    
    # Skip detection if too soon (adaptive frame skipping)
    if time_since_last < modules.globals.DETECTION_INTERVAL and FACE_DETECTION_CACHE:
        return FACE_DETECTION_CACHE.get('faces')
    
    # Perform detection
    LAST_DETECTION_TIME = current_time
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None
    
    # Cache results
    FACE_DETECTION_CACHE['faces'] = faces
    FACE_DETECTION_CACHE['timestamp'] = current_time
    
    return faces
# --- END: Mac M1-M5 Optimized Face Detection ---

# --- START: Helper function for interpolation and sharpening ---
def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    """Applies sharpening and interpolation with Apple Silicon optimizations."""
    processed_frame = current_frame

    # 1. Apply Sharpening (if enabled) with optimized kernel for Apple Silicon
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            # Ensure bbox is iterable and has 4 elements
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                # print(f"Warning: Invalid bbox format for sharpening: {bbox}") # Debug
                continue
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are integers and within bounds
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                # print(f"Warning: Could not convert bbox coordinates to int: {bbox}") # Debug
                continue


            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0: continue

            # Apply sharpening (GPU-accelerated when CUDA OpenCV is available)
            try:
                sigma = 2 if IS_APPLE_SILICON else 3
                sharpened_region = gpu_sharpen(face_region, strength=sharpness_value, sigma=sigma)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass


    # 2. Apply Interpolation (if enabled)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)

    final_frame = processed_frame # Start with the current (potentially sharpened) frame
    prev_frame = _get_previous_frame()

    if enable_interpolation and 0 < interpolation_weight < 1:
        if prev_frame is not None and prev_frame.shape == processed_frame.shape and prev_frame.dtype == processed_frame.dtype:
            # Perform interpolation
            try:
                 final_frame = gpu_add_weighted(
                    prev_frame, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight,
                    0
                 )
                 # Ensure final frame is uint8
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error as interp_e:
                 # print(f"Warning: OpenCV error during interpolation: {interp_e}") # Debug
                 final_frame = processed_frame # Use current frame if interpolation fails
                 _set_previous_frame(None) # Reset state if error occurs

            # Update the state for the next frame *with the interpolated result*
            _set_previous_frame(final_frame.copy())
        else:
            # If previous frame invalid or doesn't match, use current frame and update state
            _set_previous_frame(processed_frame.copy())
    else:
         # Interpolation is off or weight is invalid — no need to cache
         _set_previous_frame(None)


    return final_frame
# --- END: Helper function for interpolation and sharpening ---


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    """
    DEPRECATED / SIMPLER VERSION - Processes a single frame using one source face.
    Consider using process_frame_v2 for more complex scenarios.
    """
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        _set_previous_frame(None)
        return temp_frame

    # Color correction removed from here (better applied before swap if needed)

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            for face in many_faces:
                if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                    swapped_face_bboxes.append(face.bbox.astype(int))
            if len(many_faces) >= 2:
                source_faces = [source_face] * len(many_faces)
                processed_frame = batch_swap_faces(source_faces, many_faces, processed_frame)
            else:
                processed_frame = swap_face(source_face, many_faces[0], processed_frame)
    else:
        target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                    swapped_face_bboxes.append(target_face.bbox.astype(int))

    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def _build_pairs_from_file_map(temp_frame_path: str) -> list:
    """Build (source_face, target_face) pairs from source_target_map for image/video files."""
    pairs = []
    source_target_map = _MAP_STORE.get_entries()
    if not source_target_map:
        return pairs

    if modules.globals.many_faces:
        source_face = default_source_face()
        if not source_face:
            return pairs
        for map_data in source_target_map:
            if is_image(modules.globals.target_path):
                target_face = map_data.get("target", {}).get("face")
                if target_face:
                    pairs.append((source_face, target_face))
            elif is_video(modules.globals.target_path):
                for frame_data in map_data.get("target_faces_in_frame", []):
                    if frame_data and frame_data.get("location") == temp_frame_path:
                        for target_face in frame_data.get("faces", []):
                            pairs.append((source_face, target_face))
    else:
        for map_data in source_target_map:
            source_face = map_data.get("source", {}).get("face")
            if not source_face:
                continue
            if is_image(modules.globals.target_path):
                target_face = map_data.get("target", {}).get("face")
                if target_face:
                    pairs.append((source_face, target_face))
            elif is_video(modules.globals.target_path):
                for frame_data in map_data.get("target_faces_in_frame", []):
                    if frame_data and frame_data.get("location") == temp_frame_path:
                        for target_face in frame_data.get("faces", []):
                            pairs.append((source_face, target_face))
    return pairs


def _build_pairs_live(processed_frame: Frame) -> list:
    """Build (source_face, target_face) pairs for live/webcam mode."""
    pairs = []
    detected_faces = get_faces_optimized(processed_frame)
    if not detected_faces:
        return pairs

    simple_map = _MAP_STORE.get_simple_map()

    if modules.globals.many_faces:
        source_face = default_source_face()
        if source_face:
            for target_face in detected_faces:
                pairs.append((source_face, target_face))
    elif simple_map:
        source_faces = simple_map.get("source_faces", [])
        target_embeddings = simple_map.get("target_embeddings", [])
        if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
            if len(detected_faces) <= len(target_embeddings):
                for detected_face in detected_faces:
                    if detected_face.normed_embedding is None:
                        continue
                    closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                    if 0 <= closest_idx < len(source_faces):
                        pairs.append((source_faces[closest_idx], detected_face))
            else:
                detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                if not detected_embeddings:
                    return pairs
                for i, target_embedding in enumerate(target_embeddings):
                    if 0 <= i < len(source_faces):
                        closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                        if 0 <= closest_idx < len(detected_faces_with_embedding):
                            pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
    else:
        source_face = default_source_face()
        # Reuse already-detected faces instead of re-running detection.
        # get_one_face() returns the leftmost face (min bbox x), so replicate
        # that selection from the existing detection results.
        target_face = min(detected_faces, key=lambda x: x.bbox[0]) if detected_faces else None
        if source_face and target_face:
            pairs.append((source_face, target_face))
    return pairs


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Handles complex mapping scenarios (map_faces=True) and live streams."""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        _set_previous_frame(None)
        return temp_frame

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    # Determine source/target pairs based on mode
    is_file_target = modules.globals.target_path and (
        is_image(modules.globals.target_path) or is_video(modules.globals.target_path)
    )

    if is_file_target:
        source_target_pairs = _build_pairs_from_file_map(temp_frame_path)
    else:
        source_target_pairs = _build_pairs_live(processed_frame)


    # Perform swaps based on the collected pairs
    valid_pairs = [(s, t) for s, t in source_target_pairs if s and t]
    for _, target_face in valid_pairs:
        if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
            swapped_face_bboxes.append(target_face.bbox.astype(int))

    if len(valid_pairs) >= 2:
        source_list = [s for s, _ in valid_pairs]
        target_list = [t for _, t in valid_pairs]
        processed_frame = batch_swap_faces(source_list, target_list, processed_frame)
    elif len(valid_pairs) == 1:
        source_face, target_face = valid_pairs[0]
        processed_frame = swap_face(source_face, target_face, processed_frame)


    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def _load_source_face(source_path: str):
    """Load a source image and extract the primary face, logging errors.

    Returns the face object, or None when the image is missing / unreadable /
    contains no detectable face.
    """
    if not source_path or not os.path.exists(source_path):
        update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
        return None
    try:
        source_img = cv2.imread(source_path)
        if source_img is None:
            update_status(f"Error reading source image file {source_path}. Please check the path and file integrity.", NAME)
            return None
        face = get_one_face(source_img)
        if face is None:
            update_status(f"Warning: No face detected in source image {source_path}. Swaps will be skipped.", NAME)
        return face
    except Exception as e:
        import traceback
        print(f"{NAME}: Caught exception during source image processing for {source_path}:")
        traceback.print_exc()
        update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
        return None


def _process_single_frame(
    temp_frame_path: str,
    use_v2: bool,
    source_face,
    progress: Any,
) -> None:
    """Read one frame file, apply swapping, write the result back, update progress."""
    try:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            return
    except Exception as read_e:
        print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
        if progress:
            progress.update(1)
        return

    try:
        result_frame = (
            process_frame_v2(temp_frame, temp_frame_path)
            if use_v2
            else process_frame(source_face, temp_frame)
        )
        if result_frame is None:
            print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
            result_frame = temp_frame
    except Exception as proc_e:
        print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
        result_frame = temp_frame

    try:
        if not cv2.imwrite(temp_frame_path, result_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
    except Exception as write_e:
        print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")

    if progress:
        progress.update(1)


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Process a list of frame paths for video — read, swap, write back."""
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None

    if not use_v2:
        source_face = _load_source_face(source_path)
        if source_face is None:
            update_status("Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
            remaining = len(temp_frame_paths) - (progress.n if progress and hasattr(progress, 'n') else 0)
            if progress and remaining > 0:
                progress.update(remaining)
            return

    for temp_frame_path in temp_frame_paths:
        _process_single_frame(temp_frame_path, use_v2, source_face, progress)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Processes a single target image."""
    # Reset per-thread interpolation state for single image processing
    _set_previous_frame(None)

    use_v2 = getattr(modules.globals, "map_faces", False)

    # Read target first
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            # V2 processes based on global maps, doesn't need source_path here directly
            # Assumes maps are pre-populated. Pass target_path for map lookup.
            result = process_frame_v2(target_frame, target_path)

        else: # Simple mode
            try:
                source_img = cv2.imread(source_path)
                if source_img is None:
                    update_status(f"Error: Could not read source image: {source_path}", NAME)
                    return
                source_face = get_one_face(source_img)
                if not source_face:
                    update_status(f"Error: No face found in source image: {source_path}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return

            result = process_frame(source_face, target_frame)

        # Write the result if processing was successful
        if result is not None:
            write_success = cv2.imwrite(output_path, result)
            if write_success:
                update_status(f"Output image saved to: {output_path}", NAME)
            else:
                update_status(f"Error: Failed to write output image to {output_path}", NAME)
        else:
            # This case might occur if process_frame/v2 returns None unexpectedly
            update_status("Image processing failed (result was None).", NAME)

    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)
         # import traceback
         # traceback.print_exc()


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Sets up and calls the frame processing for video."""
    # Reset per-thread interpolation state before starting video processing
    _set_previous_frame(None)

    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status(f"Processing video with {mode_desc} mode.", NAME)

    # Pass the correct source_path (needed for simple mode in process_frames)
    # The core processing logic handles calling the right frame function (process_frames)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames # Pass the newly modified process_frames
    )


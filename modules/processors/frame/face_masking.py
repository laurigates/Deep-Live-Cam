import cv2
import numpy as np
from modules.typing import Face, Frame
import modules.globals
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize, gpu_cvt_color
from modules.processing_config import ProcessingConfig
from modules.processing_config_factory import build_config_from_globals

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image using LAB color space.
    Uses float32 throughout for performance (sufficient precision for 8-bit images).
    """
    if source is None or target is None:
        return source
    if source.size == 0 or target.size == 0:
        return source

    # Convert to float32 [0,1] range for proper LAB conversion
    source_f32 = source.astype(np.float32) / 255.0
    target_f32 = target.astype(np.float32) / 255.0

    source_lab = cv2.cvtColor(source_f32, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_f32, cv2.COLOR_BGR2LAB)

    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    # Reshape mean and std to be broadcastable (already float64 from meanStdDev, cast to f32)
    source_mean = source_mean.reshape(1, 1, 3).astype(np.float32)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1e-6).astype(np.float32)
    target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
    target_std = target_std.reshape(1, 1, 3).astype(np.float32)

    # Perform the color transfer in LAB space
    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

    # Convert back to BGR and uint8
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)


def apply_histogram_matching(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply histogram matching from target to source image in LAB color space.

    Performs per-channel CDF matching, which produces more aggressive correction
    than LAB mean/std transfer.  Particularly effective for cross-skin-tone swaps
    where source and target complexions differ significantly.

    Args:
        source: Swapped face crop (BGR uint8).
        target: Aligned target crop to match colours against (BGR uint8).

    Returns:
        Colour-corrected crop as BGR uint8.
    """
    if source is None or target is None:
        return source
    if source.size == 0 or target.size == 0:
        return source

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    result_channels = []
    for ch in range(3):
        src_ch = source_lab[:, :, ch]
        tgt_ch = target_lab[:, :, ch]

        # Unique pixel values and their counts — avoids building a full 256-bin
        # histogram when the image has many identical values (e.g. synthetic crops).
        src_vals, src_inv, src_cnt = np.unique(
            src_ch.ravel(), return_inverse=True, return_counts=True
        )
        tgt_vals, tgt_cnt = np.unique(tgt_ch.ravel(), return_counts=True)

        # Normalised CDFs
        src_cdf = src_cnt.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]
        tgt_cdf = tgt_cnt.cumsum().astype(np.float64)
        tgt_cdf /= tgt_cdf[-1]

        # For each source CDF value, interpolate the target pixel value whose
        # CDF is closest — this is the standard "histogram specification" mapping.
        mapped = np.interp(src_cdf, tgt_cdf, tgt_vals.astype(np.float64))
        result_channels.append(
            mapped[src_inv].reshape(src_ch.shape).clip(0, 255).astype(np.uint8)
        )

    result_lab = np.stack(result_channels, axis=2)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def create_face_mask(face: Face, frame: Frame, config=None) -> np.ndarray:
    """Creates a feathered mask covering the whole face area based on landmarks."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if face is None or not hasattr(face, 'landmark_2d_106') or frame is None:
        return mask

    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask

    try:
        if not np.all(np.isfinite(landmarks)):
            return mask

        landmarks_int = landmarks.astype(np.int32)
        face_outline = landmarks_int[0:33]

        # Estimate forehead points to ensure mask covers the whole face
        eyebrows = landmarks_int[33:43]
        if eyebrows.shape[0] > 0:
            chin = landmarks_int[16]
            eyebrow_center = np.mean(eyebrows, axis=0)
            up_vector = eyebrow_center - chin
            norm = np.linalg.norm(up_vector)
            if norm > 0:
                up_vector = up_vector / norm
                forehead_offset = up_vector * (norm * 1.0)
                forehead_points = eyebrows + forehead_offset
                top_center = np.mean(forehead_points, axis=0)
                forehead_points = (forehead_points - top_center) * 1.2 + top_center
                face_outline = np.concatenate(
                    (face_outline, forehead_points.astype(np.int32)), axis=0
                )

        try:
            hull = cv2.convexHull(face_outline.astype(np.float32))
            if hull is None or len(hull) < 3:
                return mask
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        except Exception as hull_e:
            print(f"Error creating convex hull for face mask: {hull_e}")
            return mask

        # face_mask_blur is not a standard config field — use getattr with fallback
        blur_k_size = getattr(config, 'face_mask_blur', getattr(modules.globals, "face_mask_blur", 31))
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1)
        mask = gpu_gaussian_blur(mask, (blur_k_size, blur_k_size), 0)

    except IndexError:
        pass
    except Exception as e:
        print(f"Error creating face mask: {e}")

    return mask

def create_lower_mouth_mask(
    face: Face, frame: Frame, config=None
) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0, 0, 0, 0)

    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    try:
        lower_lip_order = list(range(52, 64))

        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        if not np.all(np.isfinite(lower_lip_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        config = config or build_config_from_globals()

        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        # Correct formula: use both mask_down_size and mouth_mask_size
        expansion_factor = (
            1 + config.mask_down_size * config.mouth_mask_size
        )
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        if not np.all(np.isfinite(expanded_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add padding to the bounding box
        padding_x = int((max_x - min_x) * 0.1)
        padding_y = int((max_y - min_y) * 0.1)
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)

        if max_x > min_x and max_y > min_y:
            mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
            polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
            cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)

            # mask_blur_kernel is not a standard config field — use getattr with fallback
            blur_k_size = getattr(config, 'mask_blur_kernel', getattr(modules.globals, "mask_blur_kernel", 15))
            blur_k_size = max(1, blur_k_size // 2 * 2 + 1)
            mask_roi = gpu_gaussian_blur(mask_roi, (blur_k_size, blur_k_size), 0)

            mask[min_y:max_y, min_x:max_x] = mask_roi
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            lower_lip_polygon = expanded_landmarks
            mouth_box = (min_x, min_y, max_x, max_y)

    except IndexError:
        pass
    except Exception as e:
        print(f"Error in create_lower_mouth_mask: {e}")

    return mask, mouth_cutout, mouth_box, lower_lip_polygon

def _eye_dimensions(eye_points: np.ndarray, scale: float) -> tuple[int, int]:
    """Compute (width, height) for an eye region scaled by the mask size factor.

    Pure function — depends only on its arguments.
    """
    x_coords = eye_points[:, 0]
    y_coords = eye_points[:, 1]
    width = int((np.max(x_coords) - np.min(x_coords)) * scale)
    height = int((np.max(y_coords) - np.min(y_coords)) * scale)
    return width, height


def _ellipse_polygon(center: tuple[int, int], axes: tuple[int, int], n: int = 32) -> np.ndarray:
    """Sample *n* points on an axis-aligned ellipse for polygon visualization.

    Pure function — depends only on its arguments.
    """
    t = np.linspace(0, 2 * np.pi, n)
    x = center[0] + axes[0] * np.cos(t)
    y = center[1] + axes[1] * np.sin(t)
    return np.column_stack((x, y)).astype(np.int32)


def create_eyes_mask(face: Face, frame: Frame, config=None) -> tuple:
    config = config or build_config_from_globals()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is None:
        return mask, None, (0, 0, 0, 0), None

    left_eye = landmarks[87:96]
    right_eye = landmarks[33:42]

    left_eye_center = np.mean(left_eye, axis=0).astype(np.int32)
    right_eye_center = np.mean(right_eye, axis=0).astype(np.int32)

    scale = 1 + config.mask_down_size * config.eyes_mask_size
    left_width, left_height = _eye_dimensions(left_eye, scale)
    right_width, right_height = _eye_dimensions(right_eye, scale)

    padding = int(max(left_width, right_width) * 0.2)

    min_x = max(0, min(left_eye_center[0] - left_width // 2, right_eye_center[0] - right_width // 2) - padding)
    max_x = min(frame.shape[1], max(left_eye_center[0] + left_width // 2, right_eye_center[0] + right_width // 2) + padding)
    min_y = max(0, min(left_eye_center[1] - left_height // 2, right_eye_center[1] - right_height // 2) - padding)
    max_y = min(frame.shape[0], max(left_eye_center[1] + left_height // 2, right_eye_center[1] + right_height // 2) + padding)

    mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)

    left_center = (left_eye_center[0] - min_x, left_eye_center[1] - min_y)
    right_center = (right_eye_center[0] - min_x, right_eye_center[1] - min_y)
    left_axes = (left_width // 2, left_height // 2)
    right_axes = (right_width // 2, right_height // 2)

    cv2.ellipse(mask_roi, left_center, left_axes, 0, 0, 360, 255, -1)
    cv2.ellipse(mask_roi, right_center, right_axes, 0, 0, 360, 255, -1)
    mask_roi = gpu_gaussian_blur(mask_roi, (15, 15), 5)
    mask[min_y:max_y, min_x:max_x] = mask_roi

    eyes_cutout = frame[min_y:max_y, min_x:max_x].copy()
    left_points = _ellipse_polygon((left_eye_center[0], left_eye_center[1]), left_axes)
    right_points = _ellipse_polygon((right_eye_center[0], right_eye_center[1]), right_axes)
    eyes_polygon = np.vstack([left_points, right_points])

    return mask, eyes_cutout, (min_x, min_y, max_x, max_y), eyes_polygon

def _curved_eyebrow_contour(points: np.ndarray) -> np.ndarray:
    """Fit a smooth arch contour through eyebrow landmark points.

    Pure function — returns a new array, does not mutate input.
    Returns the input unchanged when fewer than 5 points are provided.
    """
    if len(points) < 5:
        return points

    sorted_idx = np.argsort(points[:, 0])
    sorted_pts = points[sorted_idx]

    x_min, y_min_pt = np.min(sorted_pts, axis=0)
    x_max, y_max_pt = np.max(sorted_pts, axis=0)
    width = x_max - x_min
    height = y_max_pt - y_min_pt

    x = np.linspace(x_min, x_max, 50)
    coeffs = np.polyfit(sorted_pts[:, 0], sorted_pts[:, 1], 2)
    y = np.polyval(coeffs, x)

    top_curve = y - height * 0.5
    bottom_curve = y + height * 0.2

    end_points = 5
    start_x = np.linspace(x[0] - width * 0.15, x[0], end_points)
    end_x = np.linspace(x[-1], x[-1] + width * 0.15, end_points)
    start_curve = np.column_stack((start_x, np.linspace(bottom_curve[0], top_curve[0], end_points)))
    end_curve = np.column_stack((end_x, np.linspace(bottom_curve[-1], top_curve[-1], end_points)))

    contour = np.vstack([
        start_curve,
        np.column_stack((x, top_curve)),
        end_curve,
        np.column_stack((x[::-1], bottom_curve[::-1])),
    ])

    center = np.mean(contour, axis=0)
    return center + (contour - center) * 1.2


def create_eyebrows_mask(face: Face, frame: Frame, config=None) -> tuple:
    config = config or build_config_from_globals()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is None:
        return mask, None, (0, 0, 0, 0), None

    left_eyebrow = landmarks[97:105].astype(np.float32)
    right_eyebrow = landmarks[43:51].astype(np.float32)

    all_points = np.vstack([left_eyebrow, right_eyebrow])
    pf = config.eyebrows_mask_size
    min_x = max(0, int(np.min(all_points[:, 0]) - 25 * pf))
    min_y = max(0, int(np.min(all_points[:, 1]) - 20 * pf))
    max_x = min(frame.shape[1], int(np.max(all_points[:, 0]) + 25 * pf))
    max_y = min(frame.shape[0], int(np.max(all_points[:, 1]) + 15 * pf))

    mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    origin = np.array([min_x, min_y], dtype=np.float32)

    try:
        left_shape = _curved_eyebrow_contour(left_eyebrow - origin)
        right_shape = _curved_eyebrow_contour(right_eyebrow - origin)

        cv2.fillPoly(mask_roi, [left_shape.astype(np.int32)], 255)
        cv2.fillPoly(mask_roi, [right_shape.astype(np.int32)], 255)
        # Single Gaussian blur for natural feathering.
        # Equivalent to the previous three cascading blurs (σ=7, σ=3, σ=1)
        # whose combined sigma is √(49+9+1) ≈ 7.7.  Using (0,0) lets
        # OpenCV auto-calculate the kernel size from the sigma.
        mask_roi = gpu_gaussian_blur(mask_roi, (0, 0), 8)
        mask_roi = cv2.normalize(mask_roi, None, 0, 255, cv2.NORM_MINMAX)
        mask[min_y:max_y, min_x:max_x] = mask_roi

        eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
        eyebrows_polygon = np.vstack([
            left_shape + origin,
            right_shape + origin,
        ]).astype(np.int32)
    except Exception:
        # Fallback to simple polygons if curve fitting fails
        left_local = (left_eyebrow - origin).astype(np.int32)
        right_local = (right_eyebrow - origin).astype(np.int32)
        cv2.fillPoly(mask_roi, [left_local], 255)
        cv2.fillPoly(mask_roi, [right_local], 255)
        mask_roi = gpu_gaussian_blur(mask_roi, (21, 21), 7)
        mask[min_y:max_y, min_x:max_x] = mask_roi
        eyebrows_cutout = frame[min_y:max_y, min_x:max_x].copy()
        eyebrows_polygon = np.vstack([left_eyebrow, right_eyebrow]).astype(np.int32)

    return mask, eyebrows_cutout, (min_x, min_y, max_x, max_y), eyebrows_polygon

def apply_mask_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygon: np.ndarray,
    config=None,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or polygon is None
    ):
        return frame

    try:
        resized_cutout = gpu_resize(cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_cutout.shape:
            resized_cutout = gpu_resize(
                resized_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_area = apply_color_transfer(resized_cutout, roi)

        # Create mask for the area
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        
        # Split points for left and right parts if needed
        if len(polygon) > 50:  # Arbitrary threshold to detect if we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point] - [min_x, min_y]
            right_points = polygon[mid_point:] - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [left_points], 255)
            cv2.fillPoly(polygon_mask, [right_points], 255)
        else:
            adjusted_polygon = polygon - [min_x, min_y]
            cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        _config = config or build_config_from_globals()
        # Feather the polygon mask in a single pass.
        # Combine the initial sigma (7) with the adaptive feather amount and
        # the final smoothing sigma (1) into one equivalent sigma:
        #   σ_total = √(σ_initial² + σ_adaptive² + σ_final²)
        feather_amount = min(
            _config.mouth_feather_radius * 3,  # mouth_feather_radius * 3 = 30, preserving original cap
            box_width // _config.mask_feather_ratio,
            box_height // _config.mask_feather_ratio,
        )
        combined_sigma = (49 + feather_amount ** 2 + 1) ** 0.5
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(np.float32), (0, 0), combined_sigma
        )
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask *= np.float32(1.0 / max_val)

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi.astype(np.float32) * np.float32(1.0 / 255.0))

        combined_mask_3ch = combined_mask[:, :, np.newaxis]
        inv_mask = np.float32(1.0) - combined_mask_3ch
        blended = (
            color_corrected_area * combined_mask_3ch + roi * inv_mask
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_f32 = face_mask_roi[:, :, np.newaxis].astype(np.float32) * np.float32(1.0 / 255.0)
        face_mask_3channel = np.broadcast_to(face_mask_f32, blended.shape)
        final_blend = blended * face_mask_3channel + roi * (np.float32(1.0) - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        print(f"face_masking: blending failed, returning unmodified frame: {e}")

    return frame

def draw_mask_visualization(
    frame: Frame,
    mask_data: tuple,
    label: str,
    draw_method: str = "polygon"
) -> Frame:
    mask, cutout, (min_x, min_y, max_x, max_y), polygon = mask_data

    vis_frame = frame.copy()

    # Ensure coordinates are within frame bounds
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if draw_method == "ellipse" and len(polygon) > 50:  # For eyes
        # Split points for left and right parts
        mid_point = len(polygon) // 2
        left_points = polygon[:mid_point]
        right_points = polygon[mid_point:]
        
        try:
            # Fit ellipses to points - need at least 5 points
            if len(left_points) >= 5 and len(right_points) >= 5:
                # Convert points to the correct format for ellipse fitting
                left_points = left_points.astype(np.float32)
                right_points = right_points.astype(np.float32)
                
                # Fit ellipses
                left_ellipse = cv2.fitEllipse(left_points)
                right_ellipse = cv2.fitEllipse(right_points)
                
                # Draw the ellipses
                cv2.ellipse(vis_frame, left_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vis_frame, right_ellipse, (0, 255, 0), 2)
        except Exception as e:
            # If ellipse fitting fails, draw simple rectangles as fallback
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
            cv2.rectangle(vis_frame, 
                        (left_rect[0], left_rect[1]), 
                        (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                        (0, 255, 0), 2)
            cv2.rectangle(vis_frame,
                        (right_rect[0], right_rect[1]),
                        (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]),
                        (0, 255, 0), 2)
    else:  # For mouth and eyebrows
        # Draw the polygon
        if len(polygon) > 50:  # If we have multiple parts
            mid_point = len(polygon) // 2
            left_points = polygon[:mid_point]
            right_points = polygon[mid_point:]
            cv2.polylines(vis_frame, [left_points], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(vis_frame, [right_points], True, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.polylines(vis_frame, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    # Add label
    cv2.putText(
        vis_frame,
        label,
        (min_x, min_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return vis_frame


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame

    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box

    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except ValueError:
        return frame

    if max_x <= min_x or max_y <= min_y:
        return frame

    try:
        safe_polygon = lower_lip_polygon.copy()
        safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
        safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
        cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception as e:
        print(f"Error drawing polygon for visualization: {e}")

    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15
    try:
        cv2.putText(vis_frame, "Mouth Mask", (min_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception:
        pass

    return vis_frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
    config=None,
) -> np.ndarray:
    if (frame is None or mouth_cutout is None or mouth_box is None or
            face_mask is None or mouth_polygon is None):
        return frame
    if mouth_cutout.size == 0 or face_mask.size == 0 or len(mouth_polygon) < 3:
        return frame

    try:
        min_x, min_y, max_x, max_y = map(int, mouth_box)
        box_width = max_x - min_x
        box_height = max_y - min_y

        if box_width <= 0 or box_height <= 0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)

        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            return frame

        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame

        if roi.shape[:2] != mouth_cutout.shape[:2]:
            if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                resized_mouth_cutout = gpu_resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
            else:
                return frame
        else:
            resized_mouth_cutout = mouth_cutout

        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            return frame

        color_corrected_mouth = resized_mouth_cutout
        try:
            if (len(resized_mouth_cutout.shape) == 3 and resized_mouth_cutout.shape[2] == 3 and
                    len(roi.shape) == 3 and roi.shape[2] == 3):
                color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)
        except Exception:
            pass

        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)

        _config = config or build_config_from_globals()
        mask_feather_ratio = _config.mask_feather_ratio
        feather_base_dim = min(box_width, box_height)
        feather_amount = max(1, min(30, feather_base_dim // max(1, mask_feather_ratio)))
        kernel_size = 2 * feather_amount + 1
        feathered_polygon_mask = cv2.GaussianBlur(
            polygon_mask_roi.astype(np.float32), (kernel_size, kernel_size), 0
        )
        max_val = feathered_polygon_mask.max()
        if max_val > 1e-6:
            feathered_polygon_mask = feathered_polygon_mask / max_val
        else:
            feathered_polygon_mask.fill(0.0)

        if face_mask.dtype not in (np.float32, np.float64):
            face_mask_float = face_mask.astype(np.float32) / 255.0
        else:
            face_mask_float = face_mask.astype(np.float32)
        face_mask_roi = face_mask_float[min_y:max_y, min_x:max_x]

        combined_mask = np.minimum(feathered_polygon_mask, face_mask_roi)

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            combined_mask_3channel = combined_mask[:, :, np.newaxis].astype(np.float32)
            inv_mask = np.float32(1.0) - combined_mask_3channel
            blended_roi = (color_corrected_mouth * combined_mask_3channel + roi * inv_mask)
            frame[min_y:max_y, min_x:max_x] = blended_roi.astype(np.uint8)

    except Exception as e:
        print(f"Error applying mouth area: {e}")

    return frame
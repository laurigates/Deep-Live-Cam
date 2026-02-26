import os
import shutil
from typing import Any
import insightface
import threading

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path
from modules.platform_info import IS_APPLE_SILICON

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()
_CURRENT_DET_SIZE: tuple[int, int] = (320, 320)

# Smaller detection size for Apple Silicon live mode (~4x fewer FLOPs)
_LIVE_DET_SIZE = (160, 160) if IS_APPLE_SILICON else (320, 320)
_DEFAULT_DET_SIZE = (320, 320)


def get_face_analyser() -> Any:
    """Get face analyser with thread-safe initialization."""
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None:
                # CoreML provider fails with InsightFace detection models due to
                # dynamic output shape incompatibility, so always use CPU for face analysis
                providers = [
                    p for p in modules.globals.execution_providers
                    if p != 'CoreMLExecutionProvider'
                ] or ['CPUExecutionProvider']
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers,
                    allowed_modules=['detection', 'recognition']
                )
                FACE_ANALYSER.prepare(ctx_id=0, det_size=_CURRENT_DET_SIZE)
    return FACE_ANALYSER


def set_det_size(det_size: tuple[int, int]) -> None:
    """Recreate the face analyser with a different detection size.

    InsightFace ignores ``prepare()`` after the first call, so the only
    way to change ``det_size`` is to build a fresh ``FaceAnalysis`` instance.

    Call with ``_LIVE_DET_SIZE`` when entering live mode and
    ``_DEFAULT_DET_SIZE`` when leaving it so that image/video processing
    keeps full detection resolution.
    """
    global FACE_ANALYSER, _CURRENT_DET_SIZE

    if det_size == _CURRENT_DET_SIZE:
        return

    with FACE_ANALYSER_LOCK:
        _CURRENT_DET_SIZE = det_size
        providers = [
            p for p in modules.globals.execution_providers
            if p != 'CoreMLExecutionProvider'
        ] or ['CPUExecutionProvider']
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=det_size)


def _detect_all_faces(frame: Frame) -> list:
    return get_face_analyser().get(frame)


def get_one_face(frame: Frame) -> Any:
    face = _detect_all_faces(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return _detect_all_faces(frame)
    except IndexError:
        return None


def detect_faces(frame: Frame) -> list:
    """Return detected faces based on the current many_faces setting."""
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
        return faces if faces else []
    else:
        face = get_one_face(frame)
        return [face] if face is not None else []


def has_valid_map() -> bool:
    return any(
        "source" in face_map and "target" in face_map
        for face_map in modules.globals.source_target_map
    )


def default_source_face() -> Any:
    return next(
        (face_map['source']['face']
         for face_map in modules.globals.source_target_map
         if "source" in face_map),
        None,
    )


def simplify_maps() -> None:
    paired = [
        face_map for face_map in modules.globals.source_target_map
        if "source" in face_map and "target" in face_map
    ]
    faces = [m['source']['face'] for m in paired]
    centroids = [m['target']['face'].normed_embedding for m in paired]
    with modules.globals.MAP_LOCK:
        modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}


def add_blank_map() -> None:
    with modules.globals.MAP_LOCK:
        existing_ids = [m['id'] for m in modules.globals.source_target_map]
        next_id = (max(existing_ids) + 1) if existing_ids else 0
        modules.globals.source_target_map.append({'id': next_id})


def get_unique_faces_from_target_image() -> None:
    try:
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        new_map = [
            {
                'id': i,
                'target': {
                    'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                    'face': face,
                },
            }
            for i, face in enumerate(many_faces)
            for x_min, y_min, x_max, y_max in [face['bbox']]
        ]
        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = new_map
    except ValueError:
        return
    
    
def _extract_frame_embeddings(temp_frame_paths: list) -> list:
    """Return a list of {frame, faces, location} dicts for all frames."""
    result = []
    for i, path in enumerate(tqdm(temp_frame_paths, desc="Extracting face embeddings from frames")):
        frame = cv2.imread(path)
        faces = get_many_faces(frame)
        result.append({'frame': i, 'faces': faces, 'location': path})
    return result


def _assign_centroids(frame_face_embeddings: list, centroids) -> None:
    """Tag each face in-place with the index of its closest centroid."""
    for frame in frame_face_embeddings:
        for face in frame['faces']:
            closest_idx, _ = find_closest_centroid(centroids, face.normed_embedding)
            face['target_centroid'] = closest_idx


def _build_centroid_map(frame_face_embeddings: list, centroids) -> list:
    """Build the source_target_map skeleton grouped by centroid index."""
    return [
        {
            'id': i,
            'target_faces_in_frame': [
                {
                    'frame': frame['frame'],
                    'faces': [f for f in frame['faces'] if f['target_centroid'] == i],
                    'location': frame['location'],
                }
                for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}")
            ],
        }
        for i in range(len(centroids))
    ]


def get_unique_faces_from_target_video() -> None:
    try:
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        frame_face_embeddings = _extract_frame_embeddings(temp_frame_paths)

        all_embeddings = [
            face.normed_embedding
            for frame in frame_face_embeddings
            for face in frame['faces']
        ]
        centroids = find_cluster_centroids(all_embeddings)

        _assign_centroids(frame_face_embeddings, centroids)
        new_map = _build_centroid_map(frame_face_embeddings, centroids)

        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = new_map

        default_target_face()
    except ValueError:
        return


def _find_best_face_in_frames(frames: list):
    """Return (best_face, best_frame) with the highest det_score, or (None, None)."""
    all_scored = [
        (face, frame)
        for frame in frames
        for face in frame['faces']
    ]
    if not all_scored:
        return None, None
    return max(all_scored, key=lambda pair: pair[0]['det_score'])


def default_target_face() -> None:
    for face_map in modules.globals.source_target_map:
        best_face, best_frame = _find_best_face_in_frames(face_map['target_faces_in_frame'])
        if best_face is None:
            continue
        x_min, y_min, x_max, y_max = best_face['bbox']
        target_frame = cv2.imread(best_frame['location'])
        face_map['target'] = {
            'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
            'face': best_face,
        }


def compute_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    x1 = max(float(bbox1[0]), float(bbox2[0]))
    y1 = max(float(bbox1[1]), float(bbox2[1]))
    x2 = min(float(bbox1[2]), float(bbox2[2]))
    y2 = min(float(bbox1[3]), float(bbox2[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    area1 = (float(bbox1[2]) - float(bbox1[0])) * (float(bbox1[3]) - float(bbox1[1]))
    area2 = (float(bbox2[2]) - float(bbox2[0])) * (float(bbox2[3]) - float(bbox2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0.0 else 0.0


def compute_embedding_cosine(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings.

    InsightFace ``normed_embedding`` is already L2-normalised, so the dot
    product equals the cosine similarity directly.
    """
    return float(np.dot(emb1, emb2))


def _face_pair_similar(
    face,
    prev_face,
    iou_threshold: float,
    cosine_threshold: float,
) -> bool:
    """Return True when a single face pair passes both spatial and embedding checks."""
    if not hasattr(face, 'bbox') or face.bbox is None:
        return False
    if not hasattr(prev_face, 'bbox') or prev_face.bbox is None:
        return False
    if compute_bbox_iou(face.bbox, prev_face.bbox) < iou_threshold:
        return False
    both_have_embedding = (
        hasattr(face, 'normed_embedding') and face.normed_embedding is not None
        and hasattr(prev_face, 'normed_embedding') and prev_face.normed_embedding is not None
    )
    if both_have_embedding:
        if compute_embedding_cosine(face.normed_embedding, prev_face.normed_embedding) < cosine_threshold:
            return False
    return True


def faces_are_similar(
    faces: list,
    prev_faces: list,
    iou_threshold: float = 0.9,
    cosine_threshold: float = 0.95,
) -> bool:
    """Return True if every face in *faces* matches the corresponding face in
    *prev_faces* within the given bbox IoU and embedding cosine thresholds.

    Falls back to False (force re-enhance) when:
    - Either list is empty or None
    - Face counts differ (new or lost faces)
    - Any face lacks a bbox
    """
    if not faces or not prev_faces or len(faces) != len(prev_faces):
        return False
    return all(
        _face_pair_similar(face, prev_face, iou_threshold, cosine_threshold)
        for face, prev_face in zip(faces, prev_faces)
    )


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        centroid_dir = os.path.join(temp_directory_path, str(i))
        if os.path.isdir(centroid_dir):
            shutil.rmtree(centroid_dir)
        Path(centroid_dir).mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])
            centroid_faces = [f for f in frame['faces'] if f['target_centroid'] == i]
            for j, face in enumerate(centroid_faces):
                x_min, y_min, x_max, y_max = face['bbox']
                crop = temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(centroid_dir, f"{frame['frame']}_{j}.png"), crop)
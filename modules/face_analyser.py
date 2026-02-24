import os
import platform
import shutil
from typing import Any, Tuple
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

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()
_CURRENT_DET_SIZE: Tuple[int, int] = (320, 320)
_IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'

# Smaller detection size for Apple Silicon live mode (~4x fewer FLOPs)
_LIVE_DET_SIZE = (160, 160) if _IS_APPLE_SILICON else (320, 320)
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


def set_det_size(det_size: Tuple[int, int]) -> None:
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

def has_valid_map() -> bool:
    for face_map in modules.globals.source_target_map:
        if "source" in face_map and "target" in face_map:
            return True
    return False

def default_source_face() -> Any:
    for face_map in modules.globals.source_target_map:
        if "source" in face_map:
            return face_map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for face_map in modules.globals.source_target_map:
        if "source" in face_map and "target" in face_map:
            centroids.append(face_map['target']['face'].normed_embedding)
            faces.append(face_map['source']['face'])

    with modules.globals.MAP_LOCK:
        modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        with modules.globals.MAP_LOCK:
            max_id = -1
            if len(modules.globals.source_target_map) > 0:
                max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

            modules.globals.source_target_map.append({
                    'id' : max_id + 1
                    })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        new_map = []
        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            new_map.append({
                'id' : i,
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = new_map
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        frame_face_embeddings = []
        face_embeddings = []

        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        new_map = []
        for i in range(len(centroids)):
            new_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            new_map[i]['target_faces_in_frame'] = temp

        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = new_map

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for face_map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in face_map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        if best_face is None:
            continue

        for frame in face_map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        face_map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
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
    for face, prev_face in zip(faces, prev_faces):
        if not hasattr(face, 'bbox') or face.bbox is None:
            return False
        if not hasattr(prev_face, 'bbox') or prev_face.bbox is None:
            return False
        if compute_bbox_iou(face.bbox, prev_face.bbox) < iou_threshold:
            return False
        if (
            hasattr(face, 'normed_embedding') and face.normed_embedding is not None
            and hasattr(prev_face, 'normed_embedding') and prev_face.normed_embedding is not None
        ):
            if compute_embedding_cosine(face.normed_embedding, prev_face.normed_embedding) < cosine_threshold:
                return False
    return True


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1
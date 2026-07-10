import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis as _FaceAnalysis
from tqdm import tqdm

import modules.globals
from modules.cluster_analysis import find_closest_centroid, find_cluster_centroids
from modules.face_map_store import STORE as _MAP_STORE
from modules.platform_info import IS_APPLE_SILICON
from modules.processing_config import ProcessingConfig
from modules.typing import Frame
from modules.utilities import clean_temp, create_temp, extract_frames, get_temp_directory_path, get_temp_frame_paths

# ---------------------------------------------------------------------------
# Class-based injectable API (Issue #58)
# ---------------------------------------------------------------------------


@dataclass
class FaceAnalyserConfig:
    """Configuration for a FaceAnalyser instance."""

    providers: list[str]
    det_size: tuple[int, int] = (320, 320)


class FaceAnalyser:
    """Injectable face analyser service.

    Unlike the module-level singleton functions, each ``FaceAnalyser`` instance
    owns its own ``FaceAnalysis`` object and lock, so multiple instances with
    different providers or detection sizes can coexist in the same process.

    Class attributes ``LIVE_DET_SIZE`` and ``DEFAULT_DET_SIZE`` are the canonical
    sources for detection sizes — callers should reference these rather than
    importing the private module-level ``_LIVE_DET_SIZE``/``_DEFAULT_DET_SIZE``.

    Usage::

        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=['CPUExecutionProvider'])
        analyser = FaceAnalyser(cfg)
        face = analyser.get_one_face(frame)
    """

    # Public class-level constants (use these instead of importing private _* names)
    DEFAULT_DET_SIZE: tuple[int, int] = (320, 320)
    LIVE_DET_SIZE: tuple[int, int] = (160, 160) if IS_APPLE_SILICON else (320, 320)

    def __init__(self, config: FaceAnalyserConfig) -> None:
        self._config = config
        self._det_size = config.det_size
        self._lock = threading.Lock()
        self._inner = self._build(config.providers, config.det_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_providers(providers: list[str]) -> list[str]:
        """Remove CoreML from InsightFace sessions (dynamic-shape incompatibility)."""
        filtered = [p for p in providers if p != "CoreMLExecutionProvider"]
        return filtered or ["CPUExecutionProvider"]

    def _build(self, providers: list[str], det_size: tuple[int, int]) -> Any:
        safe_providers = self._filter_providers(providers)
        inner = _FaceAnalysis(
            name="buffalo_l",
            providers=safe_providers,
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
        )
        inner.prepare(ctx_id=0, det_size=det_size)
        return inner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_one_face(self, frame: Frame) -> Any:
        """Return the leftmost detected face, or ``None`` if none found."""
        faces = self._inner.get(frame)
        try:
            return min(faces, key=lambda x: x.bbox[0])
        except ValueError:
            return None

    def get_many_faces(self, frame: Frame) -> list:
        """Return all detected faces (empty list if none found)."""
        try:
            return list(self._inner.get(frame))
        except (IndexError, ValueError):
            return []

    def set_det_size(self, det_size: tuple[int, int]) -> None:
        """Change the detection input resolution.

        Rebuilds the underlying ``FaceAnalysis`` instance because InsightFace
        silently ignores subsequent ``prepare()`` calls on the same object.
        Thread-safe via an instance-level lock.
        """
        if det_size == self._det_size:
            return
        with self._lock:
            if det_size == self._det_size:
                return  # re-check after acquiring lock
            self._det_size = det_size
            self._inner = self._build(self._config.providers, det_size)


# ---------------------------------------------------------------------------
# Module-level singleton (legacy API — kept for backward compatibility)
# ---------------------------------------------------------------------------

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()
_CURRENT_DET_SIZE: tuple[int, int] = (320, 320)

# Smaller detection size for Apple Silicon live mode (~4x fewer FLOPs)
_LIVE_DET_SIZE = (160, 160) if IS_APPLE_SILICON else (320, 320)
_DEFAULT_DET_SIZE = (320, 320)


def get_face_analyser(config: ProcessingConfig | None = None) -> Any:
    """Get face analyser with thread-safe initialization.

    When *config* is provided its ``execution_providers`` are used instead of
    reading from ``modules.globals``.  Callers that pass no argument retain the
    previous behavior unchanged.
    """
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None:
                # CoreML provider fails with InsightFace detection models due to
                # dynamic output shape incompatibility, so always use CPU for face analysis
                source_providers = (
                    config.execution_providers if config is not None else modules.globals.execution_providers
                )
                providers = [p for p in source_providers if p != "CoreMLExecutionProvider"] or ["CPUExecutionProvider"]
                FACE_ANALYSER = _FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    allowed_modules=["detection", "recognition", "landmark_2d_106"],
                )
                FACE_ANALYSER.prepare(ctx_id=0, det_size=_CURRENT_DET_SIZE)
    return FACE_ANALYSER


def set_det_size(det_size: tuple[int, int], config: ProcessingConfig | None = None) -> None:
    """Recreate the face analyser with a different detection size.

    InsightFace ignores ``prepare()`` after the first call, so the only
    way to change ``det_size`` is to build a fresh ``FaceAnalysis`` instance.

    Call with ``_LIVE_DET_SIZE`` when entering live mode and
    ``_DEFAULT_DET_SIZE`` when leaving it so that image/video processing
    keeps full detection resolution.

    When *config* is provided its ``execution_providers`` are used instead of
    reading from ``modules.globals``.
    """
    global FACE_ANALYSER, _CURRENT_DET_SIZE

    if det_size == _CURRENT_DET_SIZE:
        return

    with FACE_ANALYSER_LOCK:
        _CURRENT_DET_SIZE = det_size
        source_providers = config.execution_providers if config is not None else modules.globals.execution_providers
        providers = [p for p in source_providers if p != "CoreMLExecutionProvider"] or ["CPUExecutionProvider"]
        FACE_ANALYSER = _FaceAnalysis(
            name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition", "landmark_2d_106"]
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


def get_many_faces(frame: Frame) -> list:
    try:
        return _detect_all_faces(frame)
    except IndexError:
        return []


def detect_faces(frame: Frame, config: ProcessingConfig | None = None) -> list:
    """Return detected faces based on the current many_faces setting.

    When *config* is provided its ``many_faces`` flag is used instead of
    reading from ``modules.globals``.
    """
    many_faces = config.many_faces if config is not None else modules.globals.many_faces
    if many_faces:
        faces = get_many_faces(frame)
        return faces if faces else []
    else:
        face = get_one_face(frame)
        return [face] if face is not None else []


def detect_faces_for_webcam(frame: Frame, many_faces: bool) -> dict:
    """Detect faces in a webcam frame and return a structured result dict.

    Encapsulates the many_faces branching that previously lived in the UI
    rendering thread (``ui_webcam.py``).  Callers receive a dict with:

    * ``target_face`` — the single best face (when ``many_faces=False``), or ``None``
    * ``many_faces`` — list of all faces (when ``many_faces=True``), or ``None``

    This is the canonical detection function for the webcam pipeline.
    ``ui_webcam.py`` should call this instead of calling ``get_one_face`` /
    ``get_many_faces`` directly.
    """
    if many_faces:
        faces = get_many_faces(frame)
        return {"target_face": None, "many_faces": faces if faces else []}
    else:
        face = get_one_face(frame)
        return {"target_face": face, "many_faces": None}


def has_valid_map() -> bool:
    return _MAP_STORE.has_valid_map()


def default_source_face() -> Any:
    return _MAP_STORE.default_source_face()


def simplify_maps() -> None:
    _MAP_STORE.simplify()


def add_blank_map() -> None:
    _MAP_STORE.add_blank()


def get_unique_faces_from_target_image() -> None:
    try:
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        new_map = [
            {
                "id": i,
                "target": {
                    "cv2": target_frame[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                    "face": face,
                },
            }
            for i, face in enumerate(many_faces)
            for x_min, y_min, x_max, y_max in [face["bbox"]]
        ]
        _MAP_STORE.set_entries(new_map)
    except ValueError:
        return


def _extract_frame_embeddings(temp_frame_paths: list) -> list:
    """Return a list of {frame, faces, location} dicts for all frames."""
    result = []
    for i, path in enumerate(tqdm(temp_frame_paths, desc="Extracting face embeddings from frames")):
        frame = cv2.imread(path)
        faces = get_many_faces(frame)
        result.append({"frame": i, "faces": faces, "location": path})
    return result


def _assign_centroids(frame_face_embeddings: list, centroids) -> None:
    """Tag each face in-place with the index of its closest centroid."""
    for frame in frame_face_embeddings:
        for face in frame["faces"]:
            closest_idx, _ = find_closest_centroid(centroids, face.normed_embedding)
            face["target_centroid"] = closest_idx


def _build_centroid_map(frame_face_embeddings: list, centroids) -> list:
    """Build the source_target_map skeleton grouped by centroid index."""
    return [
        {
            "id": i,
            "target_faces_in_frame": [
                {
                    "frame": frame["frame"],
                    "faces": [f for f in frame["faces"] if f["target_centroid"] == i],
                    "location": frame["location"],
                }
                for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}")
            ],
        }
        for i in range(len(centroids))
    ]


def get_unique_faces_from_target_video() -> None:
    try:
        print("Creating temp resources...")
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print("Extracting frames...")
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        frame_face_embeddings = _extract_frame_embeddings(temp_frame_paths)

        all_embeddings = [face.normed_embedding for frame in frame_face_embeddings for face in frame["faces"]]
        centroids = find_cluster_centroids(all_embeddings)

        _assign_centroids(frame_face_embeddings, centroids)
        new_map = _build_centroid_map(frame_face_embeddings, centroids)

        _MAP_STORE.set_entries(new_map)

        default_target_face()
    except ValueError:
        return


def _find_best_face_in_frames(frames: list):
    """Return (best_face, best_frame) with the highest det_score, or (None, None)."""
    all_scored = [(face, frame) for frame in frames for face in frame["faces"]]
    if not all_scored:
        return None, None
    return max(all_scored, key=lambda pair: pair[0]["det_score"])


def default_target_face() -> None:
    for face_map in _MAP_STORE.get_entries():
        best_face, best_frame = _find_best_face_in_frames(face_map["target_faces_in_frame"])
        if best_face is None:
            continue
        x_min, y_min, x_max, y_max = best_face["bbox"]
        target_frame = cv2.imread(best_frame["location"])
        face_map["target"] = {
            "cv2": target_frame[int(y_min) : int(y_max), int(x_min) : int(x_max)],
            "face": best_face,
        }


class LandmarkSmoother:
    """Exponential moving average (EMA) smoothing for face bounding boxes and keypoints.

    Applies per-face EMA across frames to produce stable, jitter-free face detection
    output in live webcam mode.  Uses embedding cosine similarity to match faces across
    frames, enabling correct per-identity state tracking even when face count changes or
    faces temporarily swap positions.

    Example usage::

        smoother = LandmarkSmoother(alpha=0.7)

        # In detection loop:
        faces = detect(frame)
        smoother.smooth(faces)   # modifies bbox/kps in-place, returns faces

    Notes:
        * ``alpha`` is the weight given to the *current* frame's detection.
          Higher ``alpha`` → more responsive, less smoothing.
          Lower ``alpha`` → smoother output, more lag on fast movement.
        * State is automatically reset for any face whose identity cosine similarity
          falls below ``IDENTITY_THRESHOLD`` (i.e., a new face entered the scene).
        * InsightFace ``Face`` objects are dict-like and support attribute assignment,
          so bbox/kps can be updated in-place without wrapping.
    """

    #: Minimum cosine similarity between consecutive frames to treat two detections
    #: as the same person and apply EMA (rather than resetting state).
    IDENTITY_THRESHOLD: float = 0.7

    def __init__(self, alpha: float = 0.7) -> None:
        self._alpha = max(0.0, min(1.0, alpha))
        # Per-face history: list of dicts with keys 'embedding', 'bbox', 'kps'
        self._state: list[dict] = []

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = max(0.0, min(1.0, value))

    def _find_match(self, face: Any) -> dict | None:
        """Return the best-matching state entry for *face*, or ``None``.

        Matching is done via embedding cosine similarity.  InsightFace
        ``normed_embedding`` is already L2-normalised, so the dot product
        equals the cosine similarity directly.  Returns ``None`` when:

        * The state history is empty.
        * The face has no ``normed_embedding``.
        * No stored identity exceeds ``IDENTITY_THRESHOLD``.

        All pairwise similarities are computed in a single BLAS matrix
        multiply (state_matrix @ embedding) rather than an O(n) Python loop.
        """
        if not self._state:
            return None
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None

        state_matrix = np.stack([e["embedding"] for e in self._state])
        sims = state_matrix @ embedding  # single BLAS call
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        return self._state[best_idx] if best_sim >= self.IDENTITY_THRESHOLD else None

    def smooth(self, faces: list) -> list:
        """Apply EMA smoothing to *faces* in-place and return the same list.

        For each face, if a matching previous-frame identity is found, the
        ``bbox`` and ``kps`` attributes are blended with the stored values::

            face.bbox = alpha * face.bbox + (1 - alpha) * prev_bbox
            face.kps  = alpha * face.kps  + (1 - alpha) * prev_kps

        When no match is found (new face or scene change), the raw detection
        values are kept and a fresh state entry is created.

        State is cleared when *faces* is empty (no faces detected).
        """
        if not faces:
            self._state = []
            return faces

        alpha = self._alpha
        new_state: list[dict] = []

        for face in faces:
            prev = self._find_match(face)

            if prev is not None:
                if face.bbox is not None and prev["bbox"] is not None:
                    face.bbox = alpha * face.bbox + (1 - alpha) * prev["bbox"]
                if face.kps is not None and prev.get("kps") is not None:
                    face.kps = alpha * face.kps + (1 - alpha) * prev["kps"]

            embedding = getattr(face, "normed_embedding", None)
            new_state.append(
                {
                    "embedding": embedding.copy() if embedding is not None else np.zeros(512, dtype=np.float32),
                    "bbox": face.bbox.copy() if face.bbox is not None else None,
                    "kps": face.kps.copy() if face.kps is not None else None,
                }
            )

        self._state = new_state
        return faces

    def reset(self) -> None:
        """Clear all per-face smoothing state."""
        self._state = []


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
    if not hasattr(face, "bbox") or face.bbox is None:
        return False
    if not hasattr(prev_face, "bbox") or prev_face.bbox is None:
        return False
    if compute_bbox_iou(face.bbox, prev_face.bbox) < iou_threshold:
        return False
    both_have_embedding = (
        hasattr(face, "normed_embedding")
        and face.normed_embedding is not None
        and hasattr(prev_face, "normed_embedding")
        and prev_face.normed_embedding is not None
    )
    return not (
        both_have_embedding
        and compute_embedding_cosine(face.normed_embedding, prev_face.normed_embedding) < cosine_threshold
    )


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
            temp_frame = cv2.imread(frame["location"])
            centroid_faces = [f for f in frame["faces"] if f["target_centroid"] == i]
            for j, face in enumerate(centroid_faces):
                x_min, y_min, x_max, y_max = face["bbox"]
                crop = temp_frame[int(y_min) : int(y_max), int(x_min) : int(x_max)]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(centroid_dir, f"{frame['frame']}_{j}.png"), crop)

"""CodeFormer face enhancer — ONNX-based face restoration at 512×512.

CodeFormer uses a transformer + VQ-codebook architecture with an adjustable
fidelity weight ``w`` (0.0 = max quality, 1.0 = max fidelity to source).
Unlike GPEN/GFPGAN which take a single image input, CodeFormer's ONNX model
expects two inputs: ``x`` (image) and ``w`` (fidelity scalar).
"""

from typing import Any, List
import os

import cv2
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.model_loader import ModelHolder
from modules.paths import MODELS_DIR
from modules.typing import Frame, Face
from modules.utilities import conditional_download, is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

NAME = "DLC.FACE-ENHANCER-CODEFORMER"
INPUT_SIZE = 512
DEFAULT_FIDELITY = 0.7
MODEL_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"
MODEL_FILE = "codeformer.onnx"

_model = ModelHolder()
_load_error_logged = False


def _load_model() -> Any:
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"{NAME}: Model not found at {model_path}. "
            "It will be downloaded automatically — please wait and try again."
        )
    print(f"{NAME}: Loading ONNX model from {model_path}")
    session = create_onnx_session(model_path)
    print(f"{NAME}: Model loaded successfully.")
    return session


def _warmup(session: Any) -> None:
    warmup_session(session)
    print(f"{NAME}: Warmup inference complete.")


def get_enhancer() -> Any:
    return _model.get(loader_fn=_load_model, warmup_fn=_warmup)


def _get_fidelity() -> float:
    return getattr(modules.globals, "codeformer_fidelity", DEFAULT_FIDELITY)


def pre_check() -> bool:
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        update_status(f"Downloading {MODEL_FILE}...", NAME)
    conditional_download(MODELS_DIR, [MODEL_URL])
    if not os.path.exists(model_path):
        update_status(f"Model not found at {model_path}. Download may have failed.", NAME)
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    global _load_error_logged
    try:
        session = get_enhancer()
    except Exception as e:
        if not _load_error_logged:
            print(f"{NAME}: {e}")
            _load_error_logged = True
        return temp_frame
    try:
        w = np.array([_get_fidelity()], dtype=np.float32)
        return enhance_face_onnx(
            temp_frame, face, session, INPUT_SIZE,
            extra_inputs={"w": w},
        )
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(
    source_face: Face | None,
    temp_frame: Frame,
    faces=None,
    live_mode: bool = False,
) -> Frame:
    if faces is None:
        faces = get_many_faces(temp_frame) if modules.globals.many_faces else [get_one_face(temp_frame)]
    for face in faces:
        if face is not None:
            temp_frame = enhance_face(temp_frame, face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    modules.processors.frame.core.process_frames_io(
        temp_frame_paths,
        process_fn=lambda frame: process_frame(None, frame),
        progress=progress,
    )


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Frame, faces=None, live_mode: bool = False) -> Frame:
    if faces is None:
        faces = get_many_faces(temp_frame) if modules.globals.many_faces else [get_one_face(temp_frame)]
    for face in faces:
        if face is not None:
            temp_frame = enhance_face(temp_frame, face)
    return temp_frame

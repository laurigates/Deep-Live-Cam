"""Factory for ONNX-based face enhancer processor modules.

Creates a complete set of frame processor functions for an ONNX enhancer,
parameterized by model name, input size, URL, filename, and an optional
extra-input function (used by CodeFormer for its fidelity weight).

Each call to ``create_onnx_enhancer_module`` returns a dict suitable for
``globals().update()`` — the caller module gets all required plugin-interface
functions as module-level attributes.
"""

from typing import Any, Callable, List, Optional
import os

import cv2

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import detect_faces
from modules.model_loader import ModelHolder
from modules.paths import MODELS_DIR
from modules.typing import Frame, Face
from modules.utilities import download_model_if_needed, is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)


def create_onnx_enhancer_module(
    name: str,
    input_size: int,
    model_url: str,
    model_file: str,
    extra_input_fn: Optional[Callable[[], dict]] = None,
) -> dict:
    """Create a complete set of frame processor functions for an ONNX enhancer.

    Returns a dict suitable for ``globals().update()`` — the caller module gets
    all required plugin-interface functions as module-level attributes.
    """
    _model = ModelHolder()
    _load_error_logged_holder = [False]

    def _load_model() -> Any:
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"{name}: Model not found at {model_path}. "
                "It will be downloaded automatically — please wait and try again."
            )
        print(f"{name}: Loading ONNX model from {model_path}")
        session = create_onnx_session(model_path)
        print(f"{name}: Model loaded successfully.")
        return session

    def _warmup(session: Any) -> None:
        warmup_session(session)
        print(f"{name}: Warmup inference complete.")

    def get_enhancer() -> Any:
        return _model.get(loader_fn=_load_model, warmup_fn=_warmup)

    def pre_check() -> bool:
        return download_model_if_needed(model_file, [model_url], name)

    def pre_start() -> bool:
        if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
            update_status("Select an image or video for target path.", name)
            return False
        return True

    def enhance_face(temp_frame: Frame, face: Face) -> Frame:
        try:
            session = get_enhancer()
        except Exception as e:
            if not _load_error_logged_holder[0]:
                print(f"{name}: {e}")
                _load_error_logged_holder[0] = True
            return temp_frame
        try:
            extra_inputs = extra_input_fn() if extra_input_fn else None
            return enhance_face_onnx(temp_frame, face, session, input_size,
                                     extra_inputs=extra_inputs)
        except Exception as e:
            print(f"{name}: Error during face enhancement: {e}")
            return temp_frame

    def process_frame(
        source_face: Face | None,
        temp_frame: Frame,
        faces=None,
        live_mode: bool = False,
    ) -> Frame:
        if faces is None:
            faces = detect_faces(temp_frame)
        for face in faces:
            if face is not None:
                temp_frame = enhance_face(temp_frame, face)
        return temp_frame

    def process_frame_v2(temp_frame: Frame, faces=None, live_mode: bool = False) -> Frame:
        return process_frame(None, temp_frame, faces=faces, live_mode=live_mode)

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
            print(f"{name}: Error: Failed to read target image {target_path}")
            return
        result_frame = process_frame(None, target_frame)
        cv2.imwrite(output_path, result_frame)
        print(f"{name}: Enhanced image saved to {output_path}")

    def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
        modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

    return {
        'NAME': name,
        'INPUT_SIZE': input_size,
        'pre_check': pre_check,
        'pre_start': pre_start,
        'get_enhancer': get_enhancer,
        'enhance_face': enhance_face,
        'process_frame': process_frame,
        'process_frame_v2': process_frame_v2,
        'process_frames': process_frames,
        'process_image': process_image,
        'process_video': process_video,
    }

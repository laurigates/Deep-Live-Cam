"""Shared pytest configuration — stub out heavy ML imports globally."""

import sys
import types
from unittest.mock import MagicMock

import numpy as np


def _make_insightface_stub():
    insightface = types.ModuleType("insightface")
    insightface_app = types.ModuleType("insightface.app")
    insightface_app_common = types.ModuleType("insightface.app.common")
    insightface_app_common.Face = object  # lightweight substitute for the type alias
    insightface.app = insightface_app
    insightface_app.common = insightface_app_common

    # insightface.app.FaceAnalysis — used by face_analyser module
    insightface_app.FaceAnalysis = MagicMock()

    # insightface.utils.face_align — used by batch_swap_faces
    insightface_utils = types.ModuleType("insightface.utils")
    insightface_utils_face_align = types.ModuleType("insightface.utils.face_align")
    insightface_utils_face_align.norm_crop2 = MagicMock(
        side_effect=lambda img, kps, size: (
            img[:size, :size, :]
            if img.shape[0] >= size and img.shape[1] >= size
            else img[: min(img.shape[0], size), : min(img.shape[1], size), :],
            np.eye(2, 3, dtype=np.float64),
        )
    )
    insightface_utils.face_align = insightface_utils_face_align
    insightface.utils = insightface_utils

    # insightface.model_zoo — used by get_face_swapper
    insightface_model_zoo = types.ModuleType("insightface.model_zoo")
    insightface_model_zoo.get_model = MagicMock(return_value=MagicMock())
    insightface.model_zoo = insightface_model_zoo

    return {
        "insightface": insightface,
        "insightface.app": insightface_app,
        "insightface.app.common": insightface_app_common,
        "insightface.utils": insightface_utils,
        "insightface.utils.face_align": insightface_utils_face_align,
        "insightface.model_zoo": insightface_model_zoo,
    }


def _stub_ml_packages():
    # insightface submodule hierarchy
    if "insightface" not in sys.modules:
        modules = _make_insightface_stub()
        for name, mod in modules.items():
            sys.modules[name] = mod

    # Stub modules.cluster_analysis so face_analyser doesn't pull in sklearn
    if "modules.cluster_analysis" not in sys.modules:
        cluster_stub = types.ModuleType("modules.cluster_analysis")
        cluster_stub.find_cluster_centroids = MagicMock(return_value=[])
        cluster_stub.find_closest_centroid = MagicMock(return_value=(0, None))
        sys.modules["modules.cluster_analysis"] = cluster_stub

    # tkinter needs TkVersion as a float
    if "tkinter" not in sys.modules:
        tk_mock = MagicMock()
        tk_mock.TkVersion = 8.6
        sys.modules["tkinter"] = tk_mock
    sys.modules.setdefault("_tkinter", MagicMock())

    for name in [
        "onnxruntime",
        "torch",
        "tensorflow",
        "gfpgan",
        "basicsr",
        "facexlib",
        "customtkinter",
    ]:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


_stub_ml_packages()

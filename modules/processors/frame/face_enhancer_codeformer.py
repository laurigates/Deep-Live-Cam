"""CodeFormer face enhancer — ONNX-based face restoration at 512x512.

CodeFormer uses a transformer + VQ-codebook architecture with an adjustable
fidelity weight (0.0 = max quality, 1.0 = max fidelity to source).
Unlike GPEN/GFPGAN which take a single image input, CodeFormer's ONNX model
expects two inputs: ``input`` (image) and ``weight`` (fidelity scalar).
"""

import numpy as np

import modules.globals
from modules.processors.frame._onnx_enhancer_factory import create_onnx_enhancer_module

DEFAULT_FIDELITY = 0.7


def _get_fidelity() -> float:
    return getattr(modules.globals, "codeformer_fidelity", DEFAULT_FIDELITY)


_ns = create_onnx_enhancer_module(
    name="DLC.FACE-ENHANCER-CODEFORMER",
    input_size=512,
    model_url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
    model_file="codeformer.onnx",
    extra_input_fn=lambda: {"weight": np.array([_get_fidelity()], dtype=np.float64)},
)
globals().update(_ns)

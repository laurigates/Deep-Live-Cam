"""GPEN-BFR-256 face enhancer — ONNX-based face restoration at 256x256."""

from modules.processors.frame._onnx_enhancer_factory import create_onnx_enhancer_module

_ns = create_onnx_enhancer_module(
    name="DLC.FACE-ENHANCER-GPEN256",
    input_size=256,
    model_url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx",
    model_file="GPEN-BFR-256.onnx",
)
globals().update(_ns)

"""GPEN-BFR-512 face enhancer — ONNX-based face restoration at 512x512."""

from modules.processors.frame._onnx_enhancer_factory import create_onnx_enhancer_module

_ns = create_onnx_enhancer_module(
    name="DLC.FACE-ENHANCER-GPEN512",
    input_size=512,
    model_url="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-512.onnx",
    model_file="GPEN-BFR-512.onnx",
)
globals().update(_ns)

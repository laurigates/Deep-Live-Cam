"""ONNX IOBinding utility for CUDA GPU buffer reuse.

Pre-allocates output buffers on GPU and reuses them across frames,
avoiding per-frame GPU malloc overhead (~5-10ms savings on CUDA).

Only benefits CUDAExecutionProvider. CoreML and CPU providers are unsupported
and gracefully return None from create_iobinding_context().
"""
import logging
from typing import Any

import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"CUDAExecutionProvider"})

# Map ONNX tensor type strings to NumPy dtypes
_ONNX_TYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
}


def supports_iobinding(session: onnxruntime.InferenceSession) -> bool:
    """Check if the session's active provider supports IOBinding."""
    providers = session.get_providers()
    return any(p in _SUPPORTED_PROVIDERS for p in providers)


class IOBindingContext:
    """Pre-allocated GPU buffer context for ONNX Runtime IOBinding.

    Binds CPU inputs per frame, runs inference with pre-allocated GPU output
    buffers, and returns results as NumPy arrays.
    """

    def __init__(self, session: onnxruntime.InferenceSession):
        self._session = session
        self._binding = session.io_binding()

        # Pre-allocate GPU output buffers from session metadata
        self._output_ort_values: list[Any] = []
        self._output_names: list[str] = []

        for output_info in session.get_outputs():
            self._output_names.append(output_info.name)
            dtype = _ONNX_TYPE_MAP.get(output_info.type, np.float32)
            ort_value = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                output_info.shape, dtype, "cuda", 0
            )
            self._output_ort_values.append(ort_value)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference with IOBinding, reusing pre-allocated GPU output buffers.

        Args:
            inputs: Dict mapping input names to NumPy arrays (CPU).

        Returns:
            List of output NumPy arrays, matching session.run() return format.
        """
        # Bind CPU inputs
        for name, tensor in inputs.items():
            self._binding.bind_cpu_input(name, tensor)

        # Bind pre-allocated GPU outputs
        for name, ort_value in zip(self._output_names, self._output_ort_values):
            self._binding.bind_ortvalue_output(name, ort_value)

        # Run inference
        self._session.run_with_iobinding(self._binding)

        # Return outputs as NumPy arrays
        return [ov.numpy() for ov in self._output_ort_values]


def create_iobinding_context(
    session: onnxruntime.InferenceSession,
) -> IOBindingContext | None:
    """Factory that creates an IOBindingContext if supported, else returns None.

    Returns None for unsupported providers or on any initialization error.
    """
    if not supports_iobinding(session):
        return None

    try:
        return IOBindingContext(session)
    except Exception as e:
        logger.warning("Failed to create IOBinding context: %s", e)
        return None

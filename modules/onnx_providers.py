"""Build ONNX Runtime provider configuration lists."""

import os
from typing import List, Optional, Tuple, Union

from modules.platform_info import IS_APPLE_SILICON

ProviderConfig = Union[str, Tuple[str, dict]]

_COREML_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "deep-live-cam", "coreml"
)

# Valid MLComputeUnits values for onnxruntime-silicon CoreML EP
_VALID_COMPUTE_UNITS = {"ALL", "CPUAndGPU", "CPUOnly"}


def build_providers_config(
    providers: List[str],
    coreml_compute_units: Optional[str] = None,
) -> List[ProviderConfig]:
    """Convert a flat list of provider names into a config list with options.

    Handles platform-specific provider options:

    - ``CoreMLExecutionProvider`` on Apple Silicon: MLProgram format with
      configurable compute units and a persistent model cache.
      ``MLComputeUnits`` defaults to ``"ALL"`` (ANE + GPU + CPU) for maximum
      efficiency; can be overridden via the ``--coreml-compute-units`` CLI flag
      or by passing *coreml_compute_units* directly (useful for testing).
    - ``TensorrtExecutionProvider`` on NVIDIA (Linux/Windows): FP16 precision
      with persistent engine caching to ``models/trt_cache/``.  On first run
      TensorRT compiles the model (30–120 s); subsequent runs load from cache.
    - All other providers pass through unchanged.

    Args:
        providers: List of ONNX Runtime provider name strings.
        coreml_compute_units: CoreML compute unit override. If ``None``, reads
            from ``modules.globals.coreml_compute_units`` (default ``"ALL"``).

    Returns:
        List of provider configs (strings or (name, options) tuples).
    """
    if coreml_compute_units is None:
        try:
            import modules.globals
            coreml_compute_units = modules.globals.coreml_compute_units
        except AttributeError:
            coreml_compute_units = "ALL"

    if coreml_compute_units not in _VALID_COMPUTE_UNITS:
        coreml_compute_units = "ALL"
    config: List[ProviderConfig] = []
    for p in providers:
        if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            os.makedirs(_COREML_CACHE_DIR, exist_ok=True)
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": coreml_compute_units,
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                    "EnableOnSubgraphs": 1,
                    "ModelCacheDirectory": _COREML_CACHE_DIR,
                },
            ))
        elif p == "TensorrtExecutionProvider":
            from modules.tensorrt_cache import build_tensorrt_provider_options
            config.append((
                "TensorrtExecutionProvider",
                build_tensorrt_provider_options(),
            ))
        else:
            config.append(p)
    return config

"""Build ONNX Runtime provider configuration lists."""

import os
from typing import List, Tuple, Union

from modules.platform_info import IS_APPLE_SILICON

ProviderConfig = Union[str, Tuple[str, dict]]

_COREML_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "deep-live-cam", "coreml"
)


def build_providers_config(providers: List[str]) -> List[ProviderConfig]:
    """Convert a flat list of provider names into a config list with options.

    Handles platform-specific provider options:

    - ``CoreMLExecutionProvider`` on Apple Silicon: MLProgram format with
      GPU compute units and a persistent model cache.
    - ``TensorrtExecutionProvider`` on NVIDIA (Linux/Windows): FP16 precision
      with persistent engine caching to ``models/trt_cache/``.  On first run
      TensorRT compiles the model (30–120 s); subsequent runs load from cache.
    - All other providers pass through unchanged.
    """
    config: List[ProviderConfig] = []
    for p in providers:
        if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            os.makedirs(_COREML_CACHE_DIR, exist_ok=True)
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
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

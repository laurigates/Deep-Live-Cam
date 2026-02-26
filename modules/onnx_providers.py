"""Build ONNX Runtime provider configuration lists."""

import os
from typing import List, Tuple, Union

from modules.platform_info import IS_APPLE_SILICON

ProviderConfig = Union[str, Tuple[str, dict]]

_COREML_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "deep-live-cam", "coreml"
)


def build_providers_config(providers: List[str]) -> List[ProviderConfig]:
    """Convert a flat list of provider names into a config list with options."""
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
        else:
            config.append(p)
    return config

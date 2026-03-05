"""TensorRT engine cache management for Deep-Live-Cam.

ONNX Runtime's ``TensorrtExecutionProvider`` compiles ONNX models into
GPU-optimised TensorRT engines on first use (30–120 s per model).  Enabling
persistent engine caching stores those compiled engines in
``models/trt_cache/`` so subsequent runs skip compilation entirely.

ONNX Runtime manages cache invalidation automatically: if the model graph or
input shapes change, a new engine is compiled and the stale cache entry is
superseded.  No manual hash tracking is required on our side.

Usage (via ``modules.onnx_providers.build_providers_config``)::

    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    providers_config = build_providers_config(providers)
    session = onnxruntime.InferenceSession(model_path, providers=providers_config)

TensorRT is only available on NVIDIA GPUs running Linux or Windows.  On
macOS (Apple Silicon / Intel) or AMD hardware, this module is safe to import
but ``IS_TENSORRT_PLATFORM`` will be ``False``, and the EP will simply not
appear in ``onnxruntime.get_available_providers()``.
"""

import os
import sys

from modules.paths import MODELS_DIR

# Cache directory for compiled TensorRT engines
TRT_CACHE_DIR = os.path.join(MODELS_DIR, "trt_cache")

# TensorRT is only practical on NVIDIA GPUs (Linux / Windows).
# macOS does not ship the TensorRT libraries.
IS_TENSORRT_PLATFORM: bool = sys.platform in ("linux", "win32")


def get_cache_dir() -> str:
    """Return the TensorRT engine cache directory, creating it if needed.

    Returns:
        Absolute path to the ``models/trt_cache/`` directory.
    """
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    return TRT_CACHE_DIR


def has_cached_engines() -> bool:
    """Return ``True`` if compiled TensorRT engine files exist in the cache.

    This can be used to distinguish first-run (engine compilation required)
    from subsequent runs (cached engines will be loaded instantly).

    Returns:
        ``True`` when at least one ``.engine`` file is present in the cache
        directory; ``False`` otherwise.
    """
    if not os.path.exists(TRT_CACHE_DIR):
        return False
    return any(fname.endswith(".engine") for fname in os.listdir(TRT_CACHE_DIR))


def build_tensorrt_provider_options() -> dict:
    """Build the ``TensorrtExecutionProvider`` options dict.

    Enables FP16 precision and persistent engine caching to
    ``models/trt_cache/``.  On first run, ONNX Runtime compiles GPU-optimised
    TensorRT engines (30–120 s per model).  Subsequent runs load cached engines
    directly for near-instant startup.

    Returns:
        Dictionary of TensorRT execution provider options compatible with the
        ONNX Runtime session constructor's ``providers`` argument.

    Example::

        session = onnxruntime.InferenceSession(
            model_path,
            providers=[("TensorrtExecutionProvider", build_tensorrt_provider_options()),
                        "CUDAExecutionProvider"],
        )
    """
    return {
        "device_id": 0,
        "trt_fp16_enable": 1,
        "trt_engine_cache_enable": 1,
        "trt_engine_cache_path": get_cache_dir(),
        # 2 GiB workspace — sufficient for inswapper_128 and gfpgan-1024
        "trt_max_workspace_size": 2 * (1 << 30),
        # Require at least 3 ops in a subgraph before delegating to TRT;
        # avoids overhead for tiny CPU-resident subgraphs.
        "trt_min_subgraph_size": 3,
    }

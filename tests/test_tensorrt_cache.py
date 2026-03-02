"""Tests for TensorRT engine cache management — Issue #70.

Covers:
- get_cache_dir() creates the cache directory when absent.
- has_cached_engines() correctly detects presence / absence of .engine files.
- build_tensorrt_provider_options() returns required keys with correct types.
- build_providers_config() emits a (provider_name, options_dict) tuple for
  TensorrtExecutionProvider and passes other providers through unchanged.
"""

import os
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# tensorrt_cache — get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_creates_directory_when_absent(self, tmp_path):
        """get_cache_dir() must create the cache directory if it does not exist."""
        from modules import tensorrt_cache

        target = str(tmp_path / "trt_cache")
        original = tensorrt_cache.TRT_CACHE_DIR
        tensorrt_cache.TRT_CACHE_DIR = target
        try:
            result = tensorrt_cache.get_cache_dir()
            assert os.path.isdir(result)
            assert result == target
        finally:
            tensorrt_cache.TRT_CACHE_DIR = original

    def test_returns_existing_directory_without_error(self, tmp_path):
        """get_cache_dir() must succeed when the directory already exists."""
        from modules import tensorrt_cache

        target = str(tmp_path / "trt_cache")
        os.makedirs(target)
        original = tensorrt_cache.TRT_CACHE_DIR
        tensorrt_cache.TRT_CACHE_DIR = target
        try:
            result = tensorrt_cache.get_cache_dir()
            assert result == target
        finally:
            tensorrt_cache.TRT_CACHE_DIR = original


# ---------------------------------------------------------------------------
# tensorrt_cache — has_cached_engines
# ---------------------------------------------------------------------------

class TestHasCachedEngines:
    def test_returns_false_when_cache_dir_does_not_exist(self, tmp_path):
        """has_cached_engines() returns False when the cache directory is absent."""
        from modules import tensorrt_cache

        original = tensorrt_cache.TRT_CACHE_DIR
        tensorrt_cache.TRT_CACHE_DIR = str(tmp_path / "nonexistent")
        try:
            assert tensorrt_cache.has_cached_engines() is False
        finally:
            tensorrt_cache.TRT_CACHE_DIR = original

    def test_returns_false_when_no_engine_files_present(self, tmp_path):
        """has_cached_engines() returns False when directory contains no .engine files."""
        from modules import tensorrt_cache

        cache_dir = tmp_path / "trt_cache"
        cache_dir.mkdir()
        (cache_dir / "some_file.txt").write_text("dummy")
        (cache_dir / "profile.json").write_text("{}")

        original = tensorrt_cache.TRT_CACHE_DIR
        tensorrt_cache.TRT_CACHE_DIR = str(cache_dir)
        try:
            assert tensorrt_cache.has_cached_engines() is False
        finally:
            tensorrt_cache.TRT_CACHE_DIR = original

    def test_returns_true_when_engine_file_exists(self, tmp_path):
        """has_cached_engines() returns True when at least one .engine file is present."""
        from modules import tensorrt_cache

        cache_dir = tmp_path / "trt_cache"
        cache_dir.mkdir()
        (cache_dir / "inswapper_128_fp16.engine").write_bytes(b"\x00" * 8)

        original = tensorrt_cache.TRT_CACHE_DIR
        tensorrt_cache.TRT_CACHE_DIR = str(cache_dir)
        try:
            assert tensorrt_cache.has_cached_engines() is True
        finally:
            tensorrt_cache.TRT_CACHE_DIR = original


# ---------------------------------------------------------------------------
# tensorrt_cache — build_tensorrt_provider_options
# ---------------------------------------------------------------------------

class TestBuildTensorrtProviderOptions:
    def test_returns_dict(self):
        """build_tensorrt_provider_options() must return a dict."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        assert isinstance(opts, dict)

    def test_fp16_is_enabled(self):
        """FP16 precision must be enabled (trt_fp16_enable=1)."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        assert opts.get("trt_fp16_enable") == 1

    def test_engine_cache_is_enabled(self):
        """Persistent engine caching must be enabled (trt_engine_cache_enable=1)."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        assert opts.get("trt_engine_cache_enable") == 1

    def test_cache_path_is_string(self):
        """trt_engine_cache_path must be a string."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        assert isinstance(opts.get("trt_engine_cache_path"), str)

    def test_device_id_present(self):
        """device_id must be present in options."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        assert "device_id" in opts

    def test_workspace_size_is_positive_integer(self):
        """trt_max_workspace_size must be a positive integer (bytes)."""
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            opts = tensorrt_cache.build_tensorrt_provider_options()
        size = opts.get("trt_max_workspace_size", 0)
        assert isinstance(size, int) and size > 0


# ---------------------------------------------------------------------------
# onnx_providers — build_providers_config with TensorRT
# ---------------------------------------------------------------------------

class TestBuildProvidersConfigTensorRT:
    def test_tensorrt_provider_becomes_tuple_with_options(self):
        """TensorrtExecutionProvider must be converted to a (name, options) tuple."""
        from modules.onnx_providers import build_providers_config
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            config = build_providers_config(["TensorrtExecutionProvider"])

        assert len(config) == 1
        name, opts = config[0]
        assert name == "TensorrtExecutionProvider"
        assert isinstance(opts, dict)
        assert opts.get("trt_fp16_enable") == 1
        assert opts.get("trt_engine_cache_enable") == 1

    def test_tensorrt_plus_cuda_ordering(self):
        """TRT EP must come before CUDA EP in the config list (fallback chain)."""
        from modules.onnx_providers import build_providers_config
        from modules import tensorrt_cache

        with patch.object(tensorrt_cache, "get_cache_dir", return_value="/tmp/trt"):
            config = build_providers_config(
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        assert len(config) == 3
        assert config[0][0] == "TensorrtExecutionProvider"
        assert config[1] == "CUDAExecutionProvider"
        assert config[2] == "CPUExecutionProvider"

    def test_non_tensorrt_providers_pass_through_unchanged(self):
        """Providers other than TRT / CoreML must remain plain strings."""
        from modules.onnx_providers import build_providers_config

        config = build_providers_config(["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert config == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_tensorrt_options_include_cache_path(self):
        """The TRT options dict must specify the engine cache directory path."""
        from modules.onnx_providers import build_providers_config
        from modules import tensorrt_cache

        fake_cache_dir = "/tmp/fake_trt_cache"
        with patch.object(tensorrt_cache, "get_cache_dir", return_value=fake_cache_dir):
            config = build_providers_config(["TensorrtExecutionProvider"])

        _, opts = config[0]
        assert opts["trt_engine_cache_path"] == fake_cache_dir


# ---------------------------------------------------------------------------
# face_enhancer — uses build_providers_config (import check)
# ---------------------------------------------------------------------------

class TestFaceEnhancerUsesProvidersConfig:
    def test_face_enhancer_imports_build_providers_config(self):
        """face_enhancer must import build_providers_config from modules.onnx_providers."""
        import importlib
        import modules.processors.frame.face_enhancer as fe_module
        # Verify the symbol is present (the import was added)
        assert hasattr(fe_module, "build_providers_config"), (
            "face_enhancer.py must import build_providers_config from modules.onnx_providers"
        )

    def test_face_enhancer_passes_providers_through_build_config(self):
        """get_face_enhancer() must call build_providers_config before InferenceSession."""
        from modules.processors.frame import face_enhancer

        injected = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        captured_config = []

        def fake_build_providers_config(providers, **kwargs):
            captured_config.extend(providers)
            return providers  # return as-is for this test

        fake_session = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        fake_session.get_inputs.return_value = [
            __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(
                name="input", shape=[1], type="float"
            )
        ]
        fake_session.get_outputs.return_value = [
            __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(
                name="out", shape=[1], type="float"
            )
        ]
        fake_session.get_providers.return_value = injected

        with __import__("unittest.mock", fromlist=["patch"]).patch("os.path.exists", return_value=True), \
             __import__("unittest.mock", fromlist=["patch"]).patch(
                 "modules.processors.frame.face_enhancer.build_providers_config",
                 side_effect=fake_build_providers_config,
             ), \
             __import__("unittest.mock", fromlist=["patch"]).patch(
                 "modules.processors.frame.face_enhancer.onnxruntime.InferenceSession",
                 return_value=fake_session,
             ):
            original = face_enhancer.FACE_ENHANCER
            face_enhancer.FACE_ENHANCER = None
            try:
                face_enhancer.get_face_enhancer(providers=injected)
            finally:
                face_enhancer.FACE_ENHANCER = original

        assert captured_config == injected, (
            f"Expected build_providers_config to be called with {injected}, "
            f"got {captured_config}"
        )

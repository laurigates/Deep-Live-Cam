"""Tests for Phase C config injection — core.py

Verifies that start() and limit_resources() accept and use ProcessingConfig
instead of reading modules.globals directly.
"""

import inspect
from unittest.mock import MagicMock, call, patch

import pytest

from modules.processing_config import ProcessingConfig


class TestCoreFunctionSignatures:
    """Verify that core entry points accept a config parameter."""

    def test_start_accepts_config(self):
        from modules.core import start

        sig = inspect.signature(start)
        assert "config" in sig.parameters

    def test_limit_resources_accepts_config(self):
        from modules.core import limit_resources

        sig = inspect.signature(limit_resources)
        assert "config" in sig.parameters


class TestBuildConfigFromCliArgs:
    """Verify build_config_from_cli_args constructs correct ProcessingConfig."""

    def test_builds_config_with_source_and_target(self):
        from modules.processing_config_factory import build_config_from_cli_args

        args = MagicMock()
        args.source_path = "/tmp/source.jpg"
        args.target_path = "/tmp/target.mp4"
        args.output_path = "/tmp/output.mp4"
        args.frame_processor = ["face_swapper"]
        args.use_png_frames = False
        args.keep_fps = True
        args.keep_audio = True
        args.keep_frames = False
        args.many_faces = False
        args.mouth_mask = False
        args.nsfw_filter = False
        args.map_faces = False
        args.video_encoder = "libx264"
        args.video_quality = 18
        args.live_mirror = False
        args.live_resizable = False
        args.virtual_cam = False
        args.max_memory = 4
        args.execution_provider = ["cpu"]
        args.execution_threads = 4
        args.rife_enabled = False
        args.rife_model = "rife-v4.25-lite"
        args.rife_multiplier = 2
        args.half_rate_processing = False
        args.keyframe_interval = 2
        args.live_enhance_size = 256
        args.source_path_deprecated = None
        args.cpu_cores_deprecated = None
        args.gpu_vendor_deprecated = None
        args.gpu_threads_deprecated = None

        with patch("modules.core.decode_execution_providers", return_value=["CPUExecutionProvider"]):
            config = build_config_from_cli_args(args)

        assert config.source_path == "/tmp/source.jpg"
        assert config.target_path == "/tmp/target.mp4"
        assert config.frame_processors == ["face_swapper"]
        assert config.keep_fps is True
        assert config.many_faces is False

    def test_builds_config_with_many_faces(self):
        from modules.processing_config_factory import build_config_from_cli_args

        args = MagicMock()
        args.source_path = "/tmp/source.jpg"
        args.target_path = "/tmp/target.mp4"
        args.output_path = None
        args.frame_processor = ["face_swapper", "face_enhancer"]
        args.use_png_frames = True
        args.keep_fps = False
        args.keep_audio = True
        args.keep_frames = False
        args.many_faces = True
        args.mouth_mask = True
        args.nsfw_filter = False
        args.map_faces = False
        args.video_encoder = "libx265"
        args.video_quality = 22
        args.live_mirror = True
        args.live_resizable = True
        args.virtual_cam = False
        args.max_memory = 8
        args.execution_provider = ["coreml"]
        args.execution_threads = 2
        args.rife_enabled = True
        args.rife_model = "rife-v4.25"
        args.rife_multiplier = 4
        args.half_rate_processing = True
        args.keyframe_interval = 4
        args.live_enhance_size = 512
        args.source_path_deprecated = None
        args.cpu_cores_deprecated = None
        args.gpu_vendor_deprecated = None
        args.gpu_threads_deprecated = None

        with patch("modules.core.decode_execution_providers", return_value=["CoreMLExecutionProvider"]):
            config = build_config_from_cli_args(args)

        assert config.many_faces is True
        assert config.mouth_mask is True
        assert config.rife_enabled is True
        assert config.rife_multiplier == 4
        assert config.live_mirror is True
        assert isinstance(config, ProcessingConfig)

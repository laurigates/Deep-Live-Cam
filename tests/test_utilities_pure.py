"""Unit tests for pure functions in modules/utilities.py.

Tests _build_encoder_args and _build_video_ffmpeg_args which are both
documented as pure functions with no side effects.
"""

import pytest


class TestBuildEncoderArgs:
    """Tests for _build_encoder_args()."""

    def _get_fn(self):
        from modules.utilities import _build_encoder_args

        return _build_encoder_args

    # --- CUDA / NVIDIA ---

    def test_cuda_h264_uses_nvenc(self):
        fn = self._get_fn()
        encoder, opts = fn("libx264", 18, ["CUDAExecutionProvider"])
        assert encoder == "h264_nvenc"
        assert "-preset" in opts
        assert "p7" in opts
        assert "-cq" in opts
        assert "18" in opts

    def test_cuda_h265_uses_nvenc(self):
        fn = self._get_fn()
        encoder, opts = fn("libx265", 20, ["CUDAExecutionProvider"])
        assert encoder == "hevc_nvenc"

    def test_cuda_vp9_falls_through_to_cpu(self):
        """VP9 has no NVIDIA hardware encoder; should use CPU path."""
        fn = self._get_fn()
        encoder, opts = fn("libvpx-vp9", 25, ["CUDAExecutionProvider"])
        # No nvenc mapping for vp9, so it should fall through to CPU
        # Actually the function checks the mapping first, then falls through
        assert encoder == "libvpx-vp9"

    # --- DML / AMD ---

    def test_dml_h264_uses_amf(self):
        fn = self._get_fn()
        encoder, opts = fn("libx264", 18, ["DmlExecutionProvider"])
        assert encoder == "h264_amf"
        assert "-quality" in opts

    def test_dml_h265_uses_amf(self):
        fn = self._get_fn()
        encoder, opts = fn("libx265", 18, ["DmlExecutionProvider"])
        assert encoder == "hevc_amf"

    # --- CPU ---

    def test_cpu_h264_uses_preset_medium(self):
        fn = self._get_fn()
        encoder, opts = fn("libx264", 23, ["CPUExecutionProvider"])
        assert encoder == "libx264"
        assert "-preset" in opts
        assert "medium" in opts
        assert "-crf" in opts
        assert "23" in opts
        assert "-tune" in opts
        assert "film" in opts

    def test_cpu_h265_suppresses_log(self):
        fn = self._get_fn()
        encoder, opts = fn("libx265", 28, ["CPUExecutionProvider"])
        assert encoder == "libx265"
        assert "-x265-params" in opts
        assert "log-level=error" in opts

    def test_cpu_vp9(self):
        fn = self._get_fn()
        encoder, opts = fn("libvpx-vp9", 30, ["CPUExecutionProvider"])
        assert encoder == "libvpx-vp9"
        assert "-crf" in opts
        assert "30" in opts
        assert "-b:v" in opts
        assert "0" in opts

    def test_unknown_encoder_returns_empty_opts(self):
        fn = self._get_fn()
        encoder, opts = fn("unknown_codec", 18, ["CPUExecutionProvider"])
        assert encoder == "unknown_codec"
        assert opts == []

    def test_quality_none_defaults_to_18(self):
        fn = self._get_fn()
        encoder, opts = fn("libx264", None, ["CPUExecutionProvider"])
        assert "18" in opts

    def test_empty_providers_uses_cpu_path(self):
        fn = self._get_fn()
        encoder, opts = fn("libx264", 20, [])
        assert encoder == "libx264"
        assert "-crf" in opts

    def test_multiple_providers_cuda_takes_priority(self):
        """When both CUDA and CPU are listed, CUDA should be used."""
        fn = self._get_fn()
        encoder, opts = fn("libx264", 18, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert encoder == "h264_nvenc"


class TestBuildVideoFfmpegArgs:
    """Tests for _build_video_ffmpeg_args()."""

    def _get_fn(self):
        from modules.utilities import _build_video_ffmpeg_args

        return _build_video_ffmpeg_args

    def test_basic_structure(self):
        fn = self._get_fn()
        args = fn(
            fps=30.0,
            input_pattern="/tmp/frames/%04d.jpg",
            encoder="libx264",
            encoder_options=["-preset", "medium", "-crf", "18"],
            output_path="/tmp/output.mp4",
        )

        assert args[0:2] == ["-r", "30.0"]
        assert "-i" in args
        assert "/tmp/frames/%04d.jpg" in args
        assert "-c:v" in args
        assert "libx264" in args
        assert "-pix_fmt" in args
        assert "yuv420p" in args
        assert "-y" in args
        assert args[-1] == "/tmp/output.mp4"

    def test_encoder_options_included(self):
        fn = self._get_fn()
        args = fn(
            fps=24.0,
            input_pattern="in.jpg",
            encoder="h264_nvenc",
            encoder_options=["-preset", "p7", "-cq", "20"],
            output_path="out.mp4",
        )
        assert "-preset" in args
        assert "p7" in args
        assert "-cq" in args
        assert "20" in args

    def test_empty_encoder_options(self):
        fn = self._get_fn()
        args = fn(
            fps=60.0,
            input_pattern="in.jpg",
            encoder="rawvideo",
            encoder_options=[],
            output_path="out.avi",
        )
        assert "-c:v" in args
        assert "rawvideo" in args

    def test_colorspace_filter_present(self):
        fn = self._get_fn()
        args = fn(30.0, "in.jpg", "libx264", [], "out.mp4")
        assert "-vf" in args
        vf_idx = args.index("-vf")
        assert "colorspace" in args[vf_idx + 1]

    def test_faststart_flag(self):
        fn = self._get_fn()
        args = fn(30.0, "in.jpg", "libx264", [], "out.mp4")
        assert "-movflags" in args
        movflags_idx = args.index("-movflags")
        assert "+faststart" in args[movflags_idx + 1]

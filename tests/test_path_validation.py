"""Tests for path validation applied consistently to all ffmpeg subprocess calls.

Issue #90: _validate_path_for_subprocess() must guard detect_fps, extract_frames,
create_video, and restore_audio against argument injection via filenames starting with '-'.
"""
import pytest
from unittest.mock import patch, MagicMock

import modules.utilities as utilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAFE_PATH = "/tmp/video.mp4"
SAFE_OUTPUT = "/tmp/output.mp4"
UNSAFE_PATH = "/tmp/-danger.mp4"
UNSAFE_OUTPUT = "/tmp/-out.mp4"
# A path whose *dirname* starts with '-' is safe; only the basename matters.
SAFE_DIRNAME_UNSAFE_BASENAME = "/-dir/video.mp4"


# ---------------------------------------------------------------------------
# detect_fps
# ---------------------------------------------------------------------------

class TestDetectFpsValidation:
    def test_raises_for_dash_prefixed_basename(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.detect_fps(UNSAFE_PATH)

    def test_accepts_normal_path(self):
        fake_output = b"30/1\n"
        with patch("subprocess.check_output", return_value=fake_output):
            result = utilities.detect_fps(SAFE_PATH)
        assert result == pytest.approx(30.0)

    def test_accepts_path_with_dash_in_dirname(self):
        """Only the basename is checked; dashes in directory names are fine."""
        fake_output = b"25/1\n"
        with patch("subprocess.check_output", return_value=fake_output):
            result = utilities.detect_fps(SAFE_DIRNAME_UNSAFE_BASENAME)
        assert result == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

class TestExtractFramesValidation:
    def test_raises_for_dash_prefixed_basename(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.extract_frames(UNSAFE_PATH)

    def test_accepts_normal_path(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True) as mock_ffmpeg, \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/frames"):
            utilities.extract_frames(SAFE_PATH)
        mock_ffmpeg.assert_called_once()

    def test_accepts_path_with_dash_in_dirname(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True), \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/frames"):
            # Should not raise
            utilities.extract_frames(SAFE_DIRNAME_UNSAFE_BASENAME)


# ---------------------------------------------------------------------------
# create_video
# ---------------------------------------------------------------------------

class TestCreateVideoValidation:
    def test_raises_for_dash_prefixed_basename(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.create_video(UNSAFE_PATH)

    def test_accepts_normal_path(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True), \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/temp.mp4"), \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/frames"):
            utilities.create_video(SAFE_PATH, fps=30.0)

    def test_accepts_path_with_dash_in_dirname(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True), \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/temp.mp4"), \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/frames"):
            utilities.create_video(SAFE_DIRNAME_UNSAFE_BASENAME, fps=25.0)


# ---------------------------------------------------------------------------
# restore_audio — validates both target_path and output_path
# ---------------------------------------------------------------------------

class TestRestoreAudioValidation:
    def test_raises_for_dash_prefixed_target_path(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.restore_audio(UNSAFE_PATH, SAFE_OUTPUT)

    def test_raises_for_dash_prefixed_output_path(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.restore_audio(SAFE_PATH, UNSAFE_OUTPUT)

    def test_raises_for_both_unsafe(self):
        with pytest.raises(ValueError, match="Unsafe file path rejected"):
            utilities.restore_audio(UNSAFE_PATH, UNSAFE_OUTPUT)

    def test_accepts_normal_paths(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True), \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/temp.mp4"):
            utilities.restore_audio(SAFE_PATH, SAFE_OUTPUT)

    def test_accepts_paths_with_dash_in_dirname(self):
        with patch.object(utilities, "run_ffmpeg", return_value=True), \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/temp.mp4"):
            utilities.restore_audio(SAFE_DIRNAME_UNSAFE_BASENAME, "/safe-dir/output.mp4")

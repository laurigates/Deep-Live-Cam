"""Tests for video pipeline performance improvements (issues #9, #10)."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest


class TestJPEGIntermediateFrames:
    """Issue #9: Verify JPEG format is used for intermediate video frames."""

    def test_extract_frames_uses_jpg_extension(self):
        """extract_frames should write frames as .jpg files."""
        from modules import utilities

        with patch.object(utilities, "run_ffmpeg") as mock_ffmpeg, \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/test"):
            utilities.extract_frames("/fake/video.mp4")
            args = mock_ffmpeg.call_args[0][0]
            # Find the output path argument (last positional arg to ffmpeg)
            output_pattern = [a for a in args if "%04d" in a][0]
            assert output_pattern.endswith(".jpg"), f"Expected .jpg output, got {output_pattern}"
            # Verify JPEG quality flag is present
            assert "-qscale:v" in args, "Missing -qscale:v flag for JPEG quality"

    def test_get_temp_frame_paths_globs_jpg(self):
        """get_temp_frame_paths should glob for .jpg files."""
        from modules import utilities

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for ext in [".jpg", ".bmp", ".png"]:
                open(os.path.join(tmpdir, f"0001{ext}"), "w").close()

            with patch.object(utilities, "get_temp_directory_path", return_value=tmpdir):
                paths = utilities.get_temp_frame_paths("/fake/video.mp4")
                assert len(paths) == 1
                assert paths[0].endswith(".jpg")

    def test_create_video_reads_jpg_frames(self):
        """create_video should read .jpg frames as input."""
        from modules import utilities
        import modules.globals

        modules.globals.video_encoder = "libx264"
        modules.globals.video_quality = 18
        modules.globals.execution_providers = ["CPUExecutionProvider"]

        with patch.object(utilities, "run_ffmpeg") as mock_ffmpeg, \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/test/output.mp4"), \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/test"):
            utilities.create_video("/fake/video.mp4", fps=30.0)
            args = mock_ffmpeg.call_args[0][0]
            input_pattern = [a for a in args if isinstance(a, str) and "%04d" in a][0]
            assert input_pattern.endswith(".jpg"), f"Expected .jpg input, got {input_pattern}"

    def test_face_swapper_writes_jpeg_quality(self):
        """face_swapper process_frames should write with JPEG quality 95."""
        import cv2
        from unittest.mock import ANY

        with patch("cv2.imread") as mock_read, \
             patch("cv2.imwrite") as mock_write, \
             patch("modules.processors.frame.face_swapper.get_one_face") as mock_face, \
             patch("modules.processors.frame.face_swapper.process_frame") as mock_proc, \
             patch("modules.processors.frame.face_swapper.update_status"):
            import numpy as np
            fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_read.return_value = fake_frame
            mock_face.return_value = MagicMock()
            mock_proc.return_value = fake_frame

            from modules.processors.frame.face_swapper import process_frames
            # Set up globals for simple mode
            import modules.globals
            modules.globals.map_faces = False
            modules.globals.source_path = "/fake/source.jpg"

            process_frames("/fake/source.jpg", ["/tmp/0001.jpg"], progress=None)

            if mock_write.called:
                call_args = mock_write.call_args
                # Check JPEG quality params are passed
                assert call_args[0][0] == "/tmp/0001.jpg"
                assert [cv2.IMWRITE_JPEG_QUALITY, 95] == call_args[0][2]


class TestProcessPoolExecutor:
    """Issue #10: Verify ProcessPoolExecutor is used for video batch mode."""

    def test_multi_process_frame_uses_process_pool_for_video(self):
        """Video batch mode should use ProcessPoolExecutor."""
        from modules.processors.frame.core import multi_process_frame
        import modules.globals

        modules.globals.execution_threads = 2

        mock_process_frames = MagicMock()
        with patch("modules.processors.frame.core.ProcessPoolExecutor") as mock_ppe:
            mock_executor = MagicMock()
            mock_ppe.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_ppe.return_value.__exit__ = MagicMock(return_value=False)
            mock_executor.submit.return_value = MagicMock()
            mock_executor.submit.return_value.result.return_value = None

            multi_process_frame(
                "/fake/source.jpg",
                ["/tmp/0001.jpg", "/tmp/0002.jpg"],
                mock_process_frames,
                progress=None,
            )
            mock_ppe.assert_called_once()

    def test_multi_process_frame_live_uses_thread_pool(self):
        """Live mode should continue using ThreadPoolExecutor."""
        from modules.processors.frame.core import multi_process_frame_live
        import modules.globals

        modules.globals.execution_threads = 2

        mock_process_frames = MagicMock()
        with patch("modules.processors.frame.core.ThreadPoolExecutor") as mock_tpe:
            mock_executor = MagicMock()
            mock_tpe.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_tpe.return_value.__exit__ = MagicMock(return_value=False)
            mock_executor.submit.return_value = MagicMock()
            mock_executor.submit.return_value.result.return_value = None

            multi_process_frame_live(
                "/fake/source.jpg",
                ["/tmp/0001.jpg"],
                mock_process_frames,
                progress=None,
            )
            mock_tpe.assert_called_once()

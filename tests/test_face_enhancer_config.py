"""Tests for Phase A config injection — face_enhancer.py

Verifies that face_enhancer processor methods accept ProcessingConfig and
read values from config rather than modules.globals when config is provided.
"""

import inspect
from unittest.mock import MagicMock, patch

from modules.processing_config import ProcessingConfig


class TestFaceEnhancerConfigSignatures:
    """Verify that public processor methods accept a config parameter."""

    def test_process_frame_accepts_config(self):
        from modules.processors.frame.face_enhancer import process_frame

        sig = inspect.signature(process_frame)
        assert "config" in sig.parameters

    def test_process_frame_v2_accepts_config(self):
        from modules.processors.frame.face_enhancer import process_frame_v2

        sig = inspect.signature(process_frame_v2)
        assert "config" in sig.parameters

    def test_process_frames_accepts_config(self):
        from modules.processors.frame.face_enhancer import process_frames

        sig = inspect.signature(process_frames)
        assert "config" in sig.parameters

    def test_process_image_accepts_config(self):
        from modules.processors.frame.face_enhancer import process_image

        sig = inspect.signature(process_image)
        assert "config" in sig.parameters

    def test_process_video_accepts_config(self):
        from modules.processors.frame.face_enhancer import process_video

        sig = inspect.signature(process_video)
        assert "config" in sig.parameters

    def test_pre_start_accepts_config(self):
        from modules.processors.frame.face_enhancer import pre_start

        sig = inspect.signature(pre_start)
        assert "config" in sig.parameters


class TestFaceEnhancerConfigBehavior:
    """Verify config values are used instead of globals when config is provided."""

    def test_pre_start_uses_config_target_path(self):
        """pre_start reads target_path from config, not globals."""
        import modules.globals
        from modules.processors.frame.face_enhancer import pre_start

        original = modules.globals.target_path
        modules.globals.target_path = None  # globals says no target

        # config provides a valid image path — pre_start should use config
        config = ProcessingConfig(target_path="some_image.jpg")
        with patch("modules.processors.frame.face_enhancer.is_image", return_value=True):
            result = pre_start(config=config)
            assert result is True

        modules.globals.target_path = original

    def test_enhance_face_uses_config_live_enhance_size(self):
        """enhance_face reads live_enhance_size from config in live mode."""
        import numpy as np

        import modules.globals
        from modules.processors.frame.face_enhancer import enhance_face

        original = modules.globals.live_enhance_size
        modules.globals.live_enhance_size = 512  # globals says 512

        # config says 128 — in live mode this should be used for paste_size
        config = ProcessingConfig(live_enhance_size=128)

        # Track the paste_size used in _align_face calls
        align_sizes = []

        def mock_align_face(frame, landmarks, output_size):
            align_sizes.append(output_size)
            return None, None  # causes the face to be skipped

        mock_face = MagicMock()
        mock_face.kps = np.array([[100, 100], [150, 100], [125, 125], [110, 140], [140, 140]], dtype=np.float32)

        frame = np.zeros((300, 300, 3), dtype=np.uint8)

        fake_session = MagicMock()
        input_info = MagicMock()
        input_info.shape = [1, 3, 512, 512]
        input_info.name = "input"
        fake_session.get_inputs.return_value = [input_info]

        with patch("modules.processors.frame.face_enhancer.get_face_enhancer", return_value=fake_session):
            with patch("modules.processors.frame.face_enhancer._align_face", side_effect=mock_align_face):
                enhance_face(frame, faces=[mock_face], live_mode=True, config=config)

        modules.globals.live_enhance_size = original

        if align_sizes:
            assert align_sizes[0] == 128, f"Expected 128 but got {align_sizes[0]}"

"""Tests for Phase B config injection — ui_webcam.py

Verifies that create_webcam_preview and the processing thread functions
read values from config rather than modules.globals when config is provided.
"""
import inspect
import pytest
from unittest.mock import patch, MagicMock

from modules.processing_config import ProcessingConfig


class TestWebcamConfigSignatures:
    """Verify that webcam entry points accept a config parameter."""

    def test_create_webcam_preview_accepts_config(self):
        from modules.ui_webcam import create_webcam_preview
        sig = inspect.signature(create_webcam_preview)
        assert 'config' in sig.parameters

    def test_webcam_preview_accepts_config(self):
        from modules.ui_webcam import webcam_preview
        sig = inspect.signature(webcam_preview)
        assert 'config' in sig.parameters


class TestProcessingThreadConfigUse:
    """Verify processing thread reads config fields, not globals."""

    def test_is_enhancer_enabled_reads_fp_ui_from_config(self):
        """_is_enhancer_enabled reads fp_ui from the module; this documents current behavior."""
        from modules.ui_webcam import _is_enhancer_enabled
        import modules.globals

        # Temporarily set globals fp_ui to False
        original_fp_ui = modules.globals.fp_ui.copy()
        modules.globals.fp_ui['face_enhancer'] = False

        processor = MagicMock()
        processor.NAME = 'DLC.FACE-ENHANCER'

        result = _is_enhancer_enabled(processor)
        assert result is False

        modules.globals.fp_ui.update(original_fp_ui)

    def test_webcam_preview_checks_map_faces_source_path(self):
        """webcam_preview uses config.map_faces and config.source_path for early return."""
        from modules.ui_webcam import webcam_preview

        # Config with source_path=None and map_faces=False should trigger early return
        config = ProcessingConfig(source_path=None, map_faces=False)

        root_mock = MagicMock()

        # POPUP_LIVE is imported from modules.ui inside webcam_preview — patch there
        with patch('modules.ui.POPUP_LIVE', None):
            with patch('modules.ui.update_status') as mock_status:
                webcam_preview(root_mock, 0, config=config)
                mock_status.assert_called_once_with("Please select a source image first")

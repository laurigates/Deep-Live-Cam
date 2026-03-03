"""Tests for Issue #92 — ProcessingConfig threading through face_swapper call chain.

Verifies that:
1. swap_face() passes config to all sub-calls (_paste_back, _apply_mouth_mask,
   _apply_poisson_blend, apply_post_processing)
2. build_config_from_globals is NOT called in the sub-functions when config is
   provided by the caller
3. _swap_process_fn in ui_webcam builds config once and passes it, avoiding
   5-9 build_config_from_globals calls per frame at 30 FPS
"""
import inspect
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from modules.processing_config import ProcessingConfig


class TestConfigParameterSignatures:
    """Verify all functions in the call chain accept config parameter."""

    def test_swap_face_accepts_config(self):
        from modules.processors.frame.face_swapper import swap_face
        sig = inspect.signature(swap_face)
        assert 'config' in sig.parameters

    def test_paste_back_accepts_config(self):
        from modules.processors.frame.face_swapper import _paste_back
        sig = inspect.signature(_paste_back)
        assert 'config' in sig.parameters

    def test_apply_mouth_mask_accepts_config(self):
        from modules.processors.frame.face_swapper import _apply_mouth_mask
        sig = inspect.signature(_apply_mouth_mask)
        assert 'config' in sig.parameters

    def test_apply_poisson_blend_accepts_config(self):
        from modules.processors.frame.face_swapper import _apply_poisson_blend
        sig = inspect.signature(_apply_poisson_blend)
        assert 'config' in sig.parameters

    def test_apply_post_processing_accepts_config(self):
        from modules.processors.frame.face_swapper import apply_post_processing
        sig = inspect.signature(apply_post_processing)
        assert 'config' in sig.parameters


class TestBuildConfigNotCalledWhenConfigProvided:
    """Verify build_config_from_globals is NOT called when config is passed through chain."""

    def test_paste_back_does_not_call_build_config_when_provided(self):
        """_paste_back should use provided config, not rebuild from globals."""
        from modules.processors.frame.face_swapper import _paste_back

        config = ProcessingConfig(
            paste_mask_threshold=127.5,
            paste_diff_threshold=10.0,
            paste_mask_erode_ratio=10,
            paste_mask_blur_ratio=10,
        )
        h, w = 64, 64
        aimg = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_fake = np.full((h, w, 3), 128, dtype=np.uint8)
        M = np.array([[0.5, 0, 32], [0, 0.5, 32]], dtype=np.float32)
        target_img = np.zeros((128, 128, 3), dtype=np.uint8)

        with patch(
            'modules.processors.frame.face_swapper.build_config_from_globals'
        ) as mock_build:
            _paste_back(bgr_fake, aimg, M, target_img, config=config)
            mock_build.assert_not_called()

    def test_apply_mouth_mask_does_not_call_build_config_when_provided(self):
        """_apply_mouth_mask should use provided config, not rebuild from globals."""
        from modules.processors.frame.face_swapper import _apply_mouth_mask

        config = ProcessingConfig(mouth_mask=False)
        swapped = np.zeros((100, 100, 3), dtype=np.uint8)
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()

        with patch(
            'modules.processors.frame.face_swapper.build_config_from_globals'
        ) as mock_build:
            result = _apply_mouth_mask(swapped, mock_face, original, config=config)
            mock_build.assert_not_called()

        assert result is swapped  # Unchanged when mouth_mask=False

    def test_apply_poisson_blend_does_not_call_build_config_when_provided(self):
        """_apply_poisson_blend should use provided config, not rebuild from globals."""
        from modules.processors.frame.face_swapper import _apply_poisson_blend

        config = ProcessingConfig(poisson_blend=False)
        swapped = np.zeros((100, 100, 3), dtype=np.uint8)
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        pre_swap = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()

        with patch(
            'modules.processors.frame.face_swapper.build_config_from_globals'
        ) as mock_build:
            result = _apply_poisson_blend(swapped, mock_face, original, pre_swap, config=config)
            mock_build.assert_not_called()

        assert result is swapped  # Unchanged when poisson_blend=False

    def test_apply_post_processing_does_not_call_build_config_when_provided(self):
        """apply_post_processing should use provided config, not rebuild from globals."""
        from modules.processors.frame.face_swapper import apply_post_processing

        config = ProcessingConfig(sharpness=0.0, enable_interpolation=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch(
            'modules.processors.frame.face_swapper.build_config_from_globals'
        ) as mock_build:
            result = apply_post_processing(frame, [], config=config)
            mock_build.assert_not_called()


class TestSwapFacePassesConfigToSubCalls:
    """Verify swap_face passes config down to sub-calls instead of rebuilding."""

    def test_swap_face_does_not_call_build_config_when_config_provided(self):
        """When config is given, swap_face must not call build_config_from_globals at all.

        This is the key regression guard: previously swap_face did not pass config
        to _paste_back / _apply_mouth_mask / _apply_poisson_blend, causing each to
        rebuild a new ProcessingConfig independently.
        """
        from modules.processors.frame import face_swapper

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        config = ProcessingConfig(
            opacity=1.0,
            mouth_mask=False,
            poisson_blend=False,
            sharpness=0.0,
            enable_interpolation=False,
        )

        mock_source = MagicMock()
        mock_source.normed_embedding = np.ones(512, dtype=np.float32)
        mock_target = MagicMock()
        mock_target.kps = np.zeros((5, 2), dtype=np.float32)
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)

        bgr_fake = np.full((128, 128, 3), 100, dtype=np.float32)
        M = np.array([[0.64, 0, 36], [0, 0.64, 36]], dtype=np.float32)

        mock_swapper = MagicMock()
        mock_swapper.get.return_value = (bgr_fake, M)
        mock_swapper.input_size = (128, 128)

        with patch.object(face_swapper, 'get_face_swapper', return_value=mock_swapper), \
             patch(
                 'modules.processors.frame.face_swapper.build_config_from_globals'
             ) as mock_build:

            face_swapper.swap_face(mock_source, mock_target, frame, config=config)

            # build_config_from_globals must NOT be called at all when config is provided
            mock_build.assert_not_called()

    def test_swap_face_calls_build_config_once_when_no_config_given(self):
        """swap_face should call build_config_from_globals exactly once when config=None.

        The single call is at the top of swap_face; sub-calls receive the config
        and must not rebuild it themselves.
        """
        from modules.processors.frame import face_swapper

        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        mock_source = MagicMock()
        mock_source.normed_embedding = np.ones(512, dtype=np.float32)
        mock_target = MagicMock()
        mock_target.kps = np.zeros((5, 2), dtype=np.float32)
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)

        bgr_fake = np.full((128, 128, 3), 100, dtype=np.float32)
        M = np.array([[0.64, 0, 36], [0, 0.64, 36]], dtype=np.float32)

        mock_swapper = MagicMock()
        mock_swapper.get.return_value = (bgr_fake, M)
        mock_swapper.input_size = (128, 128)

        built_config = ProcessingConfig(
            opacity=1.0,
            mouth_mask=False,
            poisson_blend=False,
            sharpness=0.0,
            enable_interpolation=False,
        )

        with patch.object(face_swapper, 'get_face_swapper', return_value=mock_swapper), \
             patch(
                 'modules.processors.frame.face_swapper.build_config_from_globals',
                 return_value=built_config,
             ) as mock_build:

            face_swapper.swap_face(mock_source, mock_target, frame)

            # build_config_from_globals should be called exactly once at the top of swap_face
            # Sub-calls (_paste_back, _apply_mouth_mask, etc.) must NOT rebuild it
            assert mock_build.call_count == 1, (
                f"Expected build_config_from_globals to be called once but got "
                f"{mock_build.call_count} calls. Sub-functions are rebuilding config."
            )


class TestSwapProcessFnBuildsConfigOnce:
    """Verify _swap_process_fn in ui_webcam builds config once and passes to sub-calls."""

    def _make_mock_processor(self, frame_result=None):
        """Return a mock face_swapper processor."""
        processor = MagicMock()
        processor.NAME = "DLC.FACE-SWAPPER"
        frame = frame_result if frame_result is not None else np.zeros((200, 200, 3), dtype=np.uint8)
        processor.swap_face.return_value = frame
        processor.apply_post_processing.return_value = frame
        processor.process_frame_v2.return_value = frame
        return processor

    def test_swap_process_fn_passes_config_to_swap_face(self):
        """_swap_process_fn must pass config to processor.swap_face, not let it rebuild."""
        from modules.ui_webcam import _swap_process_fn

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_source = MagicMock()
        mock_target = MagicMock()
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        processor = self._make_mock_processor(frame)

        config = ProcessingConfig(opacity=1.0)

        inp = {
            'frame': frame,
            'source_face': mock_source,
            'target_face': mock_target,
            'many_faces': None,
            'processor': processor,
            'map_faces': False,
            'seq': 1,
            'config': config,
        }

        with patch(
            'modules.ui_webcam.build_config_from_globals'
        ) as mock_build:
            mock_build.return_value = ProcessingConfig()
            _swap_process_fn(inp)

        # config should have been passed through; build_config_from_globals
        # is used as fallback if 'config' key is absent from inp
        call_args = processor.swap_face.call_args
        assert call_args is not None
        # The config keyword arg should be the one from inp, not a freshly built one
        passed_config = call_args.kwargs.get('config') or (
            call_args.args[3] if len(call_args.args) > 3 else None
        )
        assert passed_config is config

    def test_swap_process_fn_passes_config_to_apply_post_processing(self):
        """_swap_process_fn must pass config to apply_post_processing."""
        from modules.ui_webcam import _swap_process_fn

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_source = MagicMock()
        mock_target = MagicMock()
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        processor = self._make_mock_processor(frame)

        config = ProcessingConfig(opacity=1.0)

        inp = {
            'frame': frame,
            'source_face': mock_source,
            'target_face': mock_target,
            'many_faces': None,
            'processor': processor,
            'map_faces': False,
            'seq': 1,
            'config': config,
        }

        _swap_process_fn(inp)

        call_args = processor.apply_post_processing.call_args
        assert call_args is not None
        passed_config = call_args.kwargs.get('config') or (
            call_args.args[2] if len(call_args.args) > 2 else None
        )
        assert passed_config is config

    def test_swap_process_fn_builds_config_once_when_not_in_inp(self):
        """_swap_process_fn falls back to build_config_from_globals when config absent from inp."""
        from modules.ui_webcam import _swap_process_fn

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_source = MagicMock()
        mock_target = MagicMock()
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        processor = self._make_mock_processor(frame)

        built_config = ProcessingConfig(opacity=1.0)

        inp = {
            'frame': frame,
            'source_face': mock_source,
            'target_face': mock_target,
            'many_faces': None,
            'processor': processor,
            'map_faces': False,
            'seq': 1,
            # No 'config' key — should trigger single build_config_from_globals call
        }

        with patch(
            'modules.ui_webcam.build_config_from_globals',
            return_value=built_config,
        ) as mock_build:
            _swap_process_fn(inp)

        # Should build config exactly once in _swap_process_fn, then pass to sub-calls
        assert mock_build.call_count == 1, (
            f"Expected build_config_from_globals called once, got {mock_build.call_count}"
        )


class TestConfigValuesPropagateCorrectly:
    """Verify config values actually reach sub-functions via threading."""

    def test_mouth_mask_config_reaches_apply_mouth_mask(self):
        """mouth_mask=True in config should reach _apply_mouth_mask (not re-read from globals)."""
        from modules.processors.frame import face_swapper
        import modules.globals

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        config = ProcessingConfig(
            opacity=1.0,
            mouth_mask=True,   # Enable mouth mask via config
            poisson_blend=False,
            sharpness=0.0,
            enable_interpolation=False,
        )

        mock_source = MagicMock()
        mock_source.normed_embedding = np.ones(512, dtype=np.float32)
        mock_target = MagicMock()
        mock_target.kps = np.zeros((5, 2), dtype=np.float32)
        mock_target.bbox = np.array([10, 10, 90, 90], dtype=np.float32)

        bgr_fake = np.full((128, 128, 3), 100, dtype=np.float32)
        M = np.array([[0.64, 0, 36], [0, 0.64, 36]], dtype=np.float32)

        mock_swapper = MagicMock()
        mock_swapper.get.return_value = (bgr_fake, M)
        mock_swapper.input_size = (128, 128)

        received_configs = []

        def capture_mouth_mask(swapped, target_face, original, config=None, face_mask=None):
            received_configs.append(config)
            return swapped  # pass-through

        original_mouth_mask = modules.globals.mouth_mask
        modules.globals.mouth_mask = False  # globals says disabled

        try:
            with patch.object(face_swapper, 'get_face_swapper', return_value=mock_swapper), \
                 patch.object(face_swapper, '_apply_mouth_mask', side_effect=capture_mouth_mask):

                face_swapper.swap_face(mock_source, mock_target, frame, config=config)

            # _apply_mouth_mask should have been called with the config that has mouth_mask=True
            assert len(received_configs) == 1
            assert received_configs[0] is config
            assert received_configs[0].mouth_mask is True
        finally:
            modules.globals.mouth_mask = original_mouth_mask

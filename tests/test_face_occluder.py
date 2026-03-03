"""Tests for XSeg-based face occlusion masking — Issue #72.

Verifies the face_occluder module: singleton loader, preprocessing, mask output,
fallback behavior, config integration, and pipeline integration.
"""
import inspect
import threading

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

class TestFaceOccluderSingleton:
    def test_get_face_occluder_accepts_providers_parameter(self):
        """get_face_occluder() must accept an optional providers keyword argument."""
        from modules.face_occluder import get_face_occluder
        sig = inspect.signature(get_face_occluder)
        assert 'providers' in sig.parameters

    def test_get_face_occluder_uses_injected_providers(self):
        """When providers is passed, it is forwarded to build_providers_config."""
        from modules import face_occluder

        injected = ['CPUExecutionProvider']
        captured_providers = []

        def fake_build_providers_config(providers):
            captured_providers.extend(providers)
            return providers

        fake_session = MagicMock()

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = None
        try:
            with patch('modules.face_occluder.build_providers_config',
                       side_effect=fake_build_providers_config), \
                 patch('modules.face_occluder.onnxruntime.InferenceSession',
                       return_value=fake_session), \
                 patch('modules.face_occluder.os.path.exists', return_value=True):
                face_occluder.get_face_occluder(providers=injected)
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert captured_providers == injected

    def test_get_face_occluder_falls_back_to_globals(self):
        """When no providers passed, modules.globals.execution_providers is used."""
        from modules import face_occluder
        import modules.globals

        modules.globals.execution_providers = ['CPUExecutionProvider']
        captured_providers = []

        def fake_build_providers_config(providers):
            captured_providers.extend(providers)
            return providers

        fake_session = MagicMock()

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = None
        try:
            with patch('modules.face_occluder.build_providers_config',
                       side_effect=fake_build_providers_config), \
                 patch('modules.face_occluder.onnxruntime.InferenceSession',
                       return_value=fake_session), \
                 patch('modules.face_occluder.os.path.exists', return_value=True):
                face_occluder.get_face_occluder()
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert captured_providers == ['CPUExecutionProvider']

    def test_get_face_occluder_thread_safe(self):
        """Concurrent calls to get_face_occluder() must not create multiple sessions."""
        from modules import face_occluder

        call_count = 0

        def counting_session(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock()

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = None
        try:
            with patch('modules.face_occluder.build_providers_config',
                       side_effect=lambda p: p), \
                 patch('modules.face_occluder.onnxruntime.InferenceSession',
                       side_effect=counting_session), \
                 patch('modules.face_occluder.os.path.exists', return_value=True):
                threads = [
                    threading.Thread(
                        target=face_occluder.get_face_occluder,
                        kwargs={'providers': ['CPUExecutionProvider']},
                    )
                    for _ in range(10)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert call_count == 1, f"Expected 1 session creation, got {call_count}"

    def test_get_face_occluder_returns_none_when_model_missing(self):
        """Returns None when model file does not exist and download fails."""
        from modules import face_occluder

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = None
        try:
            with patch('modules.face_occluder.os.path.exists', return_value=False), \
                 patch('modules.face_occluder.pre_check', return_value=False):
                result = face_occluder.get_face_occluder(
                    providers=['CPUExecutionProvider']
                )
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert result is None


# ---------------------------------------------------------------------------
# Preprocessing and mask output
# ---------------------------------------------------------------------------

class TestCreateOcclusionMask:
    def test_preprocessing_shape_and_dtype(self):
        """Input tensor to ONNX session must be (1, 256, 256, 3) float32 in [0, 1]."""
        from modules import face_occluder

        captured_inputs = {}

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name='input')]
        fake_session.get_inputs.return_value[0].name = 'input'
        fake_session.get_outputs.return_value = [MagicMock(name='output')]
        fake_session.get_outputs.return_value[0].name = 'output'

        def capture_run(out_names, input_dict):
            captured_inputs.update(input_dict)
            return [np.ones((1, 256, 256, 1), dtype=np.float32)]

        fake_session.run = capture_run

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = fake_session
        try:
            crop = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            face_occluder.create_occlusion_mask(crop)
        finally:
            face_occluder.FACE_OCCLUDER = original

        tensor = captured_inputs['input']
        assert tensor.shape == (1, 256, 256, 3)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_mask_output_shape_matches_input(self):
        """Output mask must have same H x W as input crop."""
        from modules import face_occluder

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name='input')]
        fake_session.get_inputs.return_value[0].name = 'input'
        fake_session.get_outputs.return_value = [MagicMock(name='output')]
        fake_session.get_outputs.return_value[0].name = 'output'
        fake_session.run = lambda out_names, input_dict: [
            np.ones((1, 256, 256, 1), dtype=np.float32)
        ]

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = fake_session
        try:
            crop = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)
            mask = face_occluder.create_occlusion_mask(crop)
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert mask.shape == (200, 150)
        assert mask.dtype == np.float32

    def test_mask_values_in_zero_one_range(self):
        """Mask values must be clipped to [0, 1]."""
        from modules import face_occluder

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name='input')]
        fake_session.get_inputs.return_value[0].name = 'input'
        fake_session.get_outputs.return_value = [MagicMock(name='output')]
        fake_session.get_outputs.return_value[0].name = 'output'
        # Return values outside [0, 1] to test clipping
        raw_output = np.full((1, 256, 256, 1), -0.5, dtype=np.float32)
        raw_output[0, :128, :, :] = 1.5
        fake_session.run = lambda out_names, input_dict: [raw_output]

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = fake_session
        try:
            crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            mask = face_occluder.create_occlusion_mask(crop)
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_fallback_all_ones_when_session_is_none(self):
        """When no model loaded, return all-ones mask (no occlusion effect)."""
        from modules import face_occluder

        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = None
        try:
            with patch.object(face_occluder, 'get_face_occluder', return_value=None):
                crop = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
                mask = face_occluder.create_occlusion_mask(crop)
        finally:
            face_occluder.FACE_OCCLUDER = original

        assert mask.shape == (100, 80)
        assert mask.dtype == np.float32
        np.testing.assert_array_equal(mask, np.ones((100, 80), dtype=np.float32))


# ---------------------------------------------------------------------------
# Occlusion mask integration (pipeline blending)
# ---------------------------------------------------------------------------

class TestOcclusionMaskIntegration:
    def test_occluded_regions_preserve_target(self):
        """Where mask is 0 (occluded), target pixels should be preserved."""
        from modules.face_occluder import create_occlusion_mask

        # Create a mask that's all zeros (fully occluded)
        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name='input')]
        fake_session.get_inputs.return_value[0].name = 'input'
        fake_session.get_outputs.return_value = [MagicMock(name='output')]
        fake_session.get_outputs.return_value[0].name = 'output'
        fake_session.run = lambda out_names, input_dict: [
            np.zeros((1, 256, 256, 1), dtype=np.float32)
        ]

        from modules import face_occluder
        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = fake_session
        try:
            bgr_fake = np.full((128, 128, 3), 255, dtype=np.float32)  # swapped (white)
            aimg = np.full((128, 128, 3), 0, dtype=np.float32)  # target (black)

            occ_mask = create_occlusion_mask(aimg.astype(np.uint8))
            occ_3ch = occ_mask[:, :, np.newaxis]
            blended = bgr_fake * occ_3ch + aimg * (1.0 - occ_3ch)
        finally:
            face_occluder.FACE_OCCLUDER = original

        # With all-zero mask, result should be all target (black)
        # Note: GaussianBlur may slightly affect edge values, so check interior
        # The mask from GaussianBlur of all-zeros is still all-zeros
        np.testing.assert_array_almost_equal(blended, aimg, decimal=1)

    def test_unoccluded_regions_use_swap(self):
        """Where mask is 1 (unoccluded), swapped pixels should be used."""
        from modules.face_occluder import create_occlusion_mask

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name='input')]
        fake_session.get_inputs.return_value[0].name = 'input'
        fake_session.get_outputs.return_value = [MagicMock(name='output')]
        fake_session.get_outputs.return_value[0].name = 'output'
        fake_session.run = lambda out_names, input_dict: [
            np.ones((1, 256, 256, 1), dtype=np.float32)
        ]

        from modules import face_occluder
        original = face_occluder.FACE_OCCLUDER
        face_occluder.FACE_OCCLUDER = fake_session
        try:
            bgr_fake = np.full((128, 128, 3), 255, dtype=np.float32)  # swapped (white)
            aimg = np.full((128, 128, 3), 0, dtype=np.float32)  # target (black)

            occ_mask = create_occlusion_mask(aimg.astype(np.uint8))
            occ_3ch = occ_mask[:, :, np.newaxis]
            blended = bgr_fake * occ_3ch + aimg * (1.0 - occ_3ch)
        finally:
            face_occluder.FACE_OCCLUDER = original

        # With all-ones mask, result should be all swapped (white)
        np.testing.assert_array_almost_equal(blended, bgr_fake, decimal=1)


# ---------------------------------------------------------------------------
# Config and globals integration
# ---------------------------------------------------------------------------

class TestOcclusionMaskConfig:
    def test_globals_has_occlusion_mask_flag(self):
        """modules.globals must have occlusion_mask attribute, default False."""
        import modules.globals
        assert hasattr(modules.globals, 'occlusion_mask')
        assert modules.globals.occlusion_mask is False

    def test_processing_config_has_occlusion_mask_field(self):
        """ProcessingConfig must have occlusion_mask field, default False."""
        from modules.processing_config import ProcessingConfig
        config = ProcessingConfig()
        assert hasattr(config, 'occlusion_mask')
        assert config.occlusion_mask is False

    def test_build_config_from_globals_maps_occlusion_mask(self):
        """build_config_from_globals must map globals.occlusion_mask to config."""
        import modules.globals
        from modules.processing_config_factory import build_config_from_globals

        original = modules.globals.occlusion_mask
        try:
            modules.globals.occlusion_mask = True
            config = build_config_from_globals()
            assert config.occlusion_mask is True

            modules.globals.occlusion_mask = False
            config = build_config_from_globals()
            assert config.occlusion_mask is False
        finally:
            modules.globals.occlusion_mask = original


# ---------------------------------------------------------------------------
# Pre-check (model download)
# ---------------------------------------------------------------------------

class TestOccluderPreCheck:
    def test_pre_check_returns_true_when_model_exists(self):
        """pre_check returns True if model file already exists."""
        from modules.face_occluder import pre_check

        with patch('modules.face_occluder.os.path.exists', return_value=True), \
             patch('modules.face_occluder.conditional_download'):
            assert pre_check() is True

    def test_pre_check_returns_false_when_download_fails(self):
        """pre_check returns False if model not found after download attempt."""
        from modules.face_occluder import pre_check

        with patch('modules.face_occluder.os.path.exists', return_value=False), \
             patch('modules.face_occluder.conditional_download'):
            assert pre_check() is False

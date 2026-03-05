"""Tests for the shared face preprocessing module (Phase 2A).

Verifies:
- preprocess_enhancement_input produces correct tensor format
- postprocess_enhancement_output reverses preprocessing
- Round-trip (postprocess(preprocess(img))) approximates original
- Delegating callers produce identical output to the shared functions
"""

import cv2
import numpy as np

from modules.face_preprocessing import (
    postprocess_enhancement_output,
    preprocess_enhancement_input,
)


class TestPreprocessEnhancementInput:
    """Tests for BGR uint8 → NCHW float32 [-1, 1] conversion."""

    def test_output_shape(self):
        face = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        assert result.shape == (1, 3, 512, 512)

    def test_output_dtype(self):
        face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        assert result.dtype == np.float32

    def test_output_range(self):
        face = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_black_maps_to_minus_one(self):
        face = np.zeros((64, 64, 3), dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

    def test_white_maps_to_one(self):
        face = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        np.testing.assert_allclose(result, 1.0, atol=1e-2)

    def test_non_square_input(self):
        face = np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)
        result = preprocess_enhancement_input(face)
        assert result.shape == (1, 3, 256, 128)


class TestPostprocessEnhancementOutput:
    """Tests for NCHW float32 [-1, 1] → BGR uint8 conversion."""

    def test_output_shape(self):
        tensor = np.random.uniform(-1, 1, (1, 3, 512, 512)).astype(np.float32)
        result = postprocess_enhancement_output(tensor)
        assert result.shape == (512, 512, 3)

    def test_output_dtype(self):
        tensor = np.random.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32)
        result = postprocess_enhancement_output(tensor)
        assert result.dtype == np.uint8

    def test_output_range(self):
        tensor = np.random.uniform(-1, 1, (1, 3, 128, 128)).astype(np.float32)
        result = postprocess_enhancement_output(tensor)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_minus_one_maps_to_black(self):
        tensor = np.full((1, 3, 64, 64), -1.0, dtype=np.float32)
        result = postprocess_enhancement_output(tensor)
        np.testing.assert_array_equal(result, 0)

    def test_one_maps_to_white(self):
        tensor = np.full((1, 3, 64, 64), 1.0, dtype=np.float32)
        result = postprocess_enhancement_output(tensor)
        np.testing.assert_array_equal(result, 255)

    def test_handles_3d_input_without_batch(self):
        """Should handle (3, H, W) input without batch dimension."""
        tensor = np.random.uniform(-1, 1, (3, 128, 128)).astype(np.float32)
        result = postprocess_enhancement_output(tensor)
        assert result.shape == (128, 128, 3)


class TestRoundTrip:
    """Verify that postprocess(preprocess(img)) ≈ original."""

    def test_round_trip_approximation(self):
        rng = np.random.RandomState(42)
        original = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        tensor = preprocess_enhancement_input(original)
        recovered = postprocess_enhancement_output(tensor)

        # Round-trip should be close (within 1 LSB due to uint8 quantisation)
        np.testing.assert_allclose(
            recovered.astype(np.float32),
            original.astype(np.float32),
            atol=1.5,
            err_msg="Round-trip should approximate original within quantisation error",
        )

    def test_round_trip_preserves_channel_order(self):
        """Ensure BGR→RGB→BGR round-trip preserves channel ordering."""
        # Create image with distinct channels
        face = np.zeros((64, 64, 3), dtype=np.uint8)
        face[:, :, 0] = 50  # B
        face[:, :, 1] = 100  # G
        face[:, :, 2] = 200  # R

        tensor = preprocess_enhancement_input(face)
        recovered = postprocess_enhancement_output(tensor)

        # Each channel should be approximately correct
        np.testing.assert_allclose(recovered[:, :, 0], 50, atol=1.5)
        np.testing.assert_allclose(recovered[:, :, 1], 100, atol=1.5)
        np.testing.assert_allclose(recovered[:, :, 2], 200, atol=1.5)


class TestDelegatingCallers:
    """Verify that face_enhancer and _onnx_enhancer produce identical output."""

    def test_face_enhancer_preprocess_matches(self):
        from modules.processors.frame.face_enhancer import _preprocess_face

        face = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        expected = preprocess_enhancement_input(face)
        actual = _preprocess_face(face)
        np.testing.assert_array_equal(actual, expected)

    def test_face_enhancer_postprocess_matches(self):
        from modules.processors.frame.face_enhancer import _postprocess_face

        tensor = np.random.uniform(-1, 1, (1, 3, 512, 512)).astype(np.float32)
        expected = postprocess_enhancement_output(tensor)
        actual = _postprocess_face(tensor)
        np.testing.assert_array_equal(actual, expected)

    def test_onnx_enhancer_preprocess_matches(self):
        from modules.processors.frame._onnx_enhancer import preprocess_face

        face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        # _onnx_enhancer resizes first, then preprocesses
        resized = cv2.resize(face, (256, 256), interpolation=cv2.INTER_LINEAR)
        expected = preprocess_enhancement_input(resized)
        actual = preprocess_face(face, 256)
        np.testing.assert_array_equal(actual, expected)

    def test_onnx_enhancer_postprocess_matches(self):
        from modules.processors.frame._onnx_enhancer import postprocess_face

        tensor = np.random.uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32)
        expected = postprocess_enhancement_output(tensor)
        actual = postprocess_face(tensor)
        np.testing.assert_array_equal(actual, expected)

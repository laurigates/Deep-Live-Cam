"""Tests for modules/processors/frame/face_enhancer.py."""
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _blank_frame(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_process_frame_v2_is_callable():
    """process_frame_v2 must exist and be callable after Wave 5 fix."""
    from modules.processors.frame import face_enhancer
    assert callable(face_enhancer.process_frame_v2)


def test_process_frame_v2_returns_frame_when_no_face():
    """process_frame_v2 must return a frame even when no face is detected."""
    frame = _blank_frame()
    with patch("modules.processors.frame.face_enhancer.get_many_faces", return_value=None):
        from modules.processors.frame import face_enhancer
        result = face_enhancer.process_frame_v2(frame)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape


def test_enhance_face_handles_runtime_error():
    """enhance_face must return the input frame when get_face_enhancer raises RuntimeError."""
    from modules.processors.frame import face_enhancer

    frame = _blank_frame()
    # Must mock get_many_faces to return a face (so enhance_face doesn't
    # early-exit), then mock get_face_enhancer to raise RuntimeError.
    mock_face = MagicMock()
    mock_face.kps = np.array([[10, 10], [20, 10], [15, 20], [10, 25], [20, 25]], dtype=np.float32)
    with patch("modules.processors.frame.face_enhancer.get_many_faces", return_value=[mock_face]):
        with patch("modules.processors.frame.face_enhancer.get_face_enhancer",
                   side_effect=RuntimeError("Model load failed")):
            result = face_enhancer.enhance_face(frame)
            assert np.array_equal(result, frame)


def test_enhance_face_uses_inter_area_for_downscale():
    """The post-inference resize should use INTER_AREA for downscaling."""
    from modules.processors.frame import face_enhancer

    # Mock face with valid landmarks
    mock_face = MagicMock()
    mock_face.kps = np.array(
        [[10, 10], [20, 10], [15, 20], [10, 25], [20, 25]], dtype=np.float32
    )

    # Mock session that outputs larger resolution than input
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 512, 512]
    mock_input.type = "tensor(float)"
    mock_session.get_inputs.return_value = [mock_input]
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_output.shape = [1, 3, 1024, 1024]
    mock_output.type = "tensor(float)"
    mock_session.get_outputs.return_value = [mock_output]
    # Return a 1024x1024 output tensor
    mock_session.run.return_value = [np.zeros((1, 3, 1024, 1024), dtype=np.float32)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch(
        "modules.processors.frame.face_enhancer.get_many_faces",
        return_value=[mock_face],
    ):
        with patch(
            "modules.processors.frame.face_enhancer.get_face_enhancer",
            return_value=mock_session,
        ):
            with patch("cv2.resize", wraps=cv2.resize) as mock_resize:
                face_enhancer.enhance_face(frame)
                # Find the resize call that does the downscale (1024->512)
                for call in mock_resize.call_args_list:
                    args, kwargs = call
                    if "interpolation" in kwargs:
                        assert kwargs["interpolation"] == cv2.INTER_AREA, (
                            f"Expected INTER_AREA, got {kwargs['interpolation']}"
                        )


# --- Feathered mask cache tests ---


def test_get_feathered_mask_shape():
    """Cached mask should have correct shape (output_size, output_size, 3)."""
    from modules.processors.frame.face_enhancer import _get_feathered_mask, _MASK_CACHE
    _MASK_CACHE.clear()
    mask = _get_feathered_mask(512)
    assert mask.shape == (512, 512, 3)
    assert mask.dtype == np.float32


def test_get_feathered_mask_cache_identity():
    """Second call should return the exact same object (cache hit)."""
    from modules.processors.frame.face_enhancer import _get_feathered_mask, _MASK_CACHE
    _MASK_CACHE.clear()
    mask1 = _get_feathered_mask(512)
    mask2 = _get_feathered_mask(512)
    assert mask1 is mask2


def test_get_feathered_mask_border_values():
    """Border pixels should be feathered (less than 1.0), center should be 1.0."""
    from modules.processors.frame.face_enhancer import _get_feathered_mask, _MASK_CACHE
    _MASK_CACHE.clear()
    mask = _get_feathered_mask(100)
    # Center should be 1.0
    assert mask[50, 50, 0] == 1.0
    # Top-left corner should be close to 0.0
    assert mask[0, 0, 0] < 0.1


def test_get_feathered_mask_immutable():
    """Cached mask should be read-only to prevent accidental modification."""
    from modules.processors.frame.face_enhancer import _get_feathered_mask, _MASK_CACHE
    _MASK_CACHE.clear()
    mask = _get_feathered_mask(256)
    assert not mask.flags.writeable


def test_get_feathered_mask_different_sizes():
    """Different sizes should produce different cache entries."""
    from modules.processors.frame.face_enhancer import _get_feathered_mask, _MASK_CACHE
    _MASK_CACHE.clear()
    mask256 = _get_feathered_mask(256)
    mask512 = _get_feathered_mask(512)
    assert mask256 is not mask512
    assert mask256.shape == (256, 256, 3)
    assert mask512.shape == (512, 512, 3)

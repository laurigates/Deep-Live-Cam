"""TDD tests for CodeFormer face enhancer processor."""

import importlib
from unittest.mock import patch

import numpy as np
import pytest

# Required interface methods for all frame processors
REQUIRED_METHODS = ["pre_check", "pre_start", "process_frame", "process_image", "process_video"]


@pytest.fixture
def codeformer_module():
    """Load the CodeFormer processor module."""
    return importlib.import_module("modules.processors.frame.face_enhancer_codeformer")


@pytest.fixture
def mock_no_face(codeformer_module):
    """Patch get_one_face to return None."""
    with patch(
        "modules.processors.frame.face_enhancer_codeformer.get_one_face",
        return_value=None,
    ):
        yield


class TestModuleInterface:
    """Verify CodeFormer module implements the frame processor interface."""

    def test_has_required_methods(self, codeformer_module):
        for method in REQUIRED_METHODS:
            assert hasattr(codeformer_module, method), f"Missing method: {method}"
            assert callable(getattr(codeformer_module, method))

    def test_has_process_frame_v2(self, codeformer_module):
        assert hasattr(codeformer_module, "process_frame_v2")
        assert callable(codeformer_module.process_frame_v2)

    def test_has_name(self, codeformer_module):
        assert hasattr(codeformer_module, "NAME")
        assert isinstance(codeformer_module.NAME, str)
        assert "CODEFORMER" in codeformer_module.NAME

    def test_input_size_is_512(self, codeformer_module):
        assert codeformer_module.INPUT_SIZE == 512


class TestNoFacePassthrough:
    """When no face is detected, the processor must return the original frame."""

    def test_process_frame_no_face_returns_original(self, codeformer_module, mock_no_face):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = codeformer_module.process_frame(None, frame)
        assert result is not None
        np.testing.assert_array_equal(result, frame)

    def test_process_frame_v2_no_face_returns_original(self, codeformer_module, mock_no_face):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = codeformer_module.process_frame_v2(frame)
        assert result is not None
        np.testing.assert_array_equal(result, frame)


class TestOutputProperties:
    """Verify output frame has correct shape and dtype."""

    def test_output_shape_matches_input(self, codeformer_module, mock_no_face):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = codeformer_module.process_frame(None, frame)
        assert result.shape == frame.shape

    def test_output_dtype_is_uint8(self, codeformer_module, mock_no_face):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = codeformer_module.process_frame(None, frame)
        assert result.dtype == np.uint8


class TestFidelityWeight:
    """CodeFormer-specific: verify fidelity weight parameter handling."""

    def test_has_default_fidelity(self, codeformer_module):
        """Module should define a default fidelity weight."""
        assert hasattr(codeformer_module, "DEFAULT_FIDELITY")
        w = codeformer_module.DEFAULT_FIDELITY
        assert 0.0 <= w <= 1.0

    def test_fidelity_from_globals(self):
        """Fidelity weight should be configurable via globals."""
        import modules.globals
        assert hasattr(modules.globals, "codeformer_fidelity")
        assert 0.0 <= modules.globals.codeformer_fidelity <= 1.0


class TestOnnxEnhancerExtraInputs:
    """Verify _onnx_enhancer supports extra_inputs for CodeFormer."""

    def test_enhance_face_onnx_accepts_extra_inputs(self):
        from modules.processors.frame._onnx_enhancer import enhance_face_onnx
        import inspect
        sig = inspect.signature(enhance_face_onnx)
        assert "extra_inputs" in sig.parameters


class TestCliRegistration:
    """Verify CodeFormer is registered as a CLI choice."""

    def test_codeformer_in_fp_ui(self):
        import modules.globals
        assert "face_enhancer_codeformer" in modules.globals.fp_ui

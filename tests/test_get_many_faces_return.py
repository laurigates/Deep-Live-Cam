"""Tests for get_many_faces return type consistency (Issue #98).

Both the module-level and class-based get_many_faces must always return a list,
never None, so callers can safely iterate without a None guard.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Module-level get_many_faces
# ---------------------------------------------------------------------------

class TestModuleLevelGetManyFaces:
    """Module-level face_analyser.get_many_faces always returns list."""

    def test_returns_list_when_no_faces_detected(self):
        """IndexError from _detect_all_faces must produce [] not None."""
        import modules.face_analyser as fa

        with patch.object(fa, '_detect_all_faces', side_effect=IndexError):
            result = fa.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == [], f"Expected [], got {result!r}"
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_returns_list_type_annotation(self):
        """Return type annotation must be list, not Any."""
        import inspect
        import modules.face_analyser as fa

        hints = fa.get_many_faces.__annotations__
        assert hints.get('return') is list, (
            f"Expected return annotation 'list', got {hints.get('return')!r}"
        )

    def test_returns_faces_list_on_success(self):
        """Successful detection returns the faces list unchanged."""
        import modules.face_analyser as fa

        fake_faces = [MagicMock(), MagicMock()]
        with patch.object(fa, '_detect_all_faces', return_value=fake_faces):
            result = fa.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result is fake_faces


# ---------------------------------------------------------------------------
# Class-based FaceAnalyser.get_many_faces
# ---------------------------------------------------------------------------

class TestClassBasedGetManyFaces:
    """FaceAnalyser.get_many_faces always returns list and propagates real errors."""

    def _make_analyser(self, inner_mock):
        """Build a FaceAnalyser with a pre-set _inner mock, bypassing __init__."""
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig
        cfg = FaceAnalyserConfig(providers=['CPUExecutionProvider'])
        analyser = object.__new__(FaceAnalyser)
        analyser._config = cfg
        analyser._det_size = cfg.det_size
        import threading
        analyser._lock = threading.Lock()
        analyser._inner = inner_mock
        return analyser

    def test_returns_empty_list_on_index_error(self):
        """IndexError from _inner.get must produce []."""
        inner = MagicMock()
        inner.get.side_effect = IndexError("empty")
        analyser = self._make_analyser(inner)

        result = analyser.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == []
        assert isinstance(result, list)

    def test_returns_empty_list_on_value_error(self):
        """ValueError from _inner.get must produce []."""
        inner = MagicMock()
        inner.get.side_effect = ValueError("no faces")
        analyser = self._make_analyser(inner)

        result = analyser.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == []
        assert isinstance(result, list)

    def test_propagates_runtime_error(self):
        """RuntimeError (e.g. ONNX inference failure) must NOT be swallowed."""
        inner = MagicMock()
        inner.get.side_effect = RuntimeError("ONNX session error")
        analyser = self._make_analyser(inner)

        with pytest.raises(RuntimeError, match="ONNX session error"):
            analyser.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_propagates_onnx_exception(self):
        """Generic Exception subclasses other than IndexError/ValueError propagate."""
        inner = MagicMock()
        inner.get.side_effect = MemoryError("OOM")
        analyser = self._make_analyser(inner)

        with pytest.raises(MemoryError):
            analyser.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_returns_face_list_on_success(self):
        """Successful detection returns list of detected faces."""
        inner = MagicMock()
        fake_faces = [MagicMock(), MagicMock()]
        inner.get.return_value = fake_faces
        analyser = self._make_analyser(inner)

        result = analyser.get_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == fake_faces
        assert isinstance(result, list)

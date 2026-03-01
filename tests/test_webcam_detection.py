"""Tests for the webcam detection abstraction — Issue #60.

Verifies that:
1. Private _LIVE_DET_SIZE/_DEFAULT_DET_SIZE are no longer necessary to import
   from outside face_analyser.py — FaceAnalyser exposes them as class attributes.
2. detect_faces_for_webcam() encapsulates the many_faces branching,
   making detection logic testable without loading the UI.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Public det size constants via FaceAnalyser class
# ---------------------------------------------------------------------------


class TestFaceAnalyserPublicDeSizes:
    def test_live_det_size_is_class_attribute(self):
        """LIVE_DET_SIZE must be a public class attribute (no underscore prefix)."""
        from modules.face_analyser import FaceAnalyser

        assert hasattr(FaceAnalyser, "LIVE_DET_SIZE")
        size = FaceAnalyser.LIVE_DET_SIZE
        assert isinstance(size, tuple)
        assert len(size) == 2

    def test_default_det_size_is_class_attribute(self):
        """DEFAULT_DET_SIZE must be a public class attribute."""
        from modules.face_analyser import FaceAnalyser

        assert hasattr(FaceAnalyser, "DEFAULT_DET_SIZE")
        size = FaceAnalyser.DEFAULT_DET_SIZE
        assert size == (320, 320)

    def test_live_det_size_is_smaller_or_equal_on_apple_silicon(self):
        """On Apple Silicon LIVE_DET_SIZE should be smaller than DEFAULT_DET_SIZE."""
        from modules.face_analyser import FaceAnalyser
        from modules.platform_info import IS_APPLE_SILICON

        if IS_APPLE_SILICON:
            assert FaceAnalyser.LIVE_DET_SIZE[0] <= FaceAnalyser.DEFAULT_DET_SIZE[0]

    def test_private_constants_still_exist_for_compat(self):
        """_LIVE_DET_SIZE and _DEFAULT_DET_SIZE must still be importable for backward compat."""
        from modules.face_analyser import _DEFAULT_DET_SIZE, _LIVE_DET_SIZE

        assert isinstance(_LIVE_DET_SIZE, tuple)
        assert isinstance(_DEFAULT_DET_SIZE, tuple)

    def test_class_attrs_match_private_constants(self):
        """Class attributes must match the module-level private constants."""
        from modules.face_analyser import _DEFAULT_DET_SIZE, _LIVE_DET_SIZE, FaceAnalyser

        assert FaceAnalyser.LIVE_DET_SIZE == _LIVE_DET_SIZE
        assert FaceAnalyser.DEFAULT_DET_SIZE == _DEFAULT_DET_SIZE


# ---------------------------------------------------------------------------
# detect_faces_for_webcam
# ---------------------------------------------------------------------------


class TestDetectFacesForWebcam:
    def test_function_exists(self):
        from modules.face_analyser import detect_faces_for_webcam

        assert callable(detect_faces_for_webcam)

    def test_many_faces_true_returns_many_faces_list(self):
        from modules.face_analyser import detect_faces_for_webcam

        frame = MagicMock()
        face_a = MagicMock()
        face_b = MagicMock()
        with patch("modules.face_analyser.get_many_faces", return_value=[face_a, face_b]):
            with patch("modules.face_analyser.get_one_face") as mock_one:
                result = detect_faces_for_webcam(frame, many_faces=True)
        mock_one.assert_not_called()
        assert result["many_faces"] == [face_a, face_b]
        assert result["target_face"] is None

    def test_many_faces_false_returns_one_face(self):
        from modules.face_analyser import detect_faces_for_webcam

        frame = MagicMock()
        face = MagicMock()
        with patch("modules.face_analyser.get_one_face", return_value=face):
            with patch("modules.face_analyser.get_many_faces") as mock_many:
                result = detect_faces_for_webcam(frame, many_faces=False)
        mock_many.assert_not_called()
        assert result["target_face"] is face
        assert result["many_faces"] is None

    def test_no_face_found_returns_none_target(self):
        from modules.face_analyser import detect_faces_for_webcam

        frame = MagicMock()
        with patch("modules.face_analyser.get_one_face", return_value=None):
            result = detect_faces_for_webcam(frame, many_faces=False)
        assert result["target_face"] is None

    def test_returns_dict_with_required_keys(self):
        from modules.face_analyser import detect_faces_for_webcam

        frame = MagicMock()
        with patch("modules.face_analyser.get_one_face", return_value=None):
            result = detect_faces_for_webcam(frame, many_faces=False)
        assert "target_face" in result
        assert "many_faces" in result

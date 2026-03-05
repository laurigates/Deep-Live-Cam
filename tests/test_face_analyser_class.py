"""Tests for the FaceAnalyser class-based API — Issue #58.

These tests verify that FaceAnalyser provides a proper class-based injectable
service, allowing multiple independent instances with different configurations.
"""

import threading
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# FaceAnalyserConfig structure
# ---------------------------------------------------------------------------


class TestFaceAnalyserConfig:
    def test_can_instantiate_with_providers(self):
        from modules.face_analyser import FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        assert cfg.providers == ["CPUExecutionProvider"]

    def test_default_det_size_is_320(self):
        from modules.face_analyser import FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        assert cfg.det_size == (320, 320)

    def test_custom_det_size(self):
        from modules.face_analyser import FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"], det_size=(160, 160))
        assert cfg.det_size == (160, 160)


# ---------------------------------------------------------------------------
# FaceAnalyser instantiation
# ---------------------------------------------------------------------------


class TestFaceAnalyserInstantiation:
    def test_can_create_instance(self):
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner):
            analyser = FaceAnalyser(cfg)
        assert analyser is not None

    def test_filters_coreml_from_insightface(self):
        """CoreML must be filtered from InsightFace (dynamic shape issue)."""
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner) as MockFA:
            FaceAnalyser(cfg)
        _, call_kwargs = MockFA.call_args
        assert "CoreMLExecutionProvider" not in call_kwargs["providers"]
        assert "CPUExecutionProvider" in call_kwargs["providers"]

    def test_falls_back_to_cpu_if_all_filtered(self):
        """If all providers are filtered, CPUExecutionProvider must be added."""
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CoreMLExecutionProvider"])
        mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner) as MockFA:
            FaceAnalyser(cfg)
        _, call_kwargs = MockFA.call_args
        assert "CPUExecutionProvider" in call_kwargs["providers"]

    def test_two_instances_are_independent(self):
        """Two FaceAnalyser instances must not share state."""
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg_a = FaceAnalyserConfig(providers=["CPUExecutionProvider"], det_size=(320, 320))
        cfg_b = FaceAnalyserConfig(providers=["CPUExecutionProvider"], det_size=(160, 160))
        mock_a = MagicMock()
        mock_b = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", side_effect=[mock_a, mock_b]):
            analyser_a = FaceAnalyser(cfg_a)
            analyser_b = FaceAnalyser(cfg_b)
        assert analyser_a is not analyser_b


# ---------------------------------------------------------------------------
# FaceAnalyser.get_one_face
# ---------------------------------------------------------------------------


class TestFaceAnalyserGetOneFace:
    def _make_analyser(self, mock_inner=None):
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        if mock_inner is None:
            mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner):
            return FaceAnalyser(cfg), mock_inner

    def test_returns_leftmost_face(self):
        face_a = MagicMock()
        face_a.bbox = [10.0, 20.0, 110.0, 120.0]
        face_b = MagicMock()
        face_b.bbox = [200.0, 20.0, 300.0, 120.0]
        mock_inner = MagicMock()
        mock_inner.get.return_value = [face_a, face_b]
        analyser, _ = self._make_analyser(mock_inner)
        result = analyser.get_one_face(MagicMock())
        assert result is face_a

    def test_returns_none_when_no_faces(self):
        mock_inner = MagicMock()
        mock_inner.get.return_value = []
        analyser, _ = self._make_analyser(mock_inner)
        result = analyser.get_one_face(MagicMock())
        assert result is None


# ---------------------------------------------------------------------------
# FaceAnalyser.get_many_faces
# ---------------------------------------------------------------------------


class TestFaceAnalyserGetManyFaces:
    def _make_analyser(self, mock_inner=None):
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        if mock_inner is None:
            mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner):
            return FaceAnalyser(cfg), mock_inner

    def test_returns_all_detected_faces(self):
        face_a = MagicMock()
        face_b = MagicMock()
        mock_inner = MagicMock()
        mock_inner.get.return_value = [face_a, face_b]
        analyser, _ = self._make_analyser(mock_inner)
        result = analyser.get_many_faces(MagicMock())
        assert result == [face_a, face_b]

    def test_returns_empty_list_when_no_faces(self):
        mock_inner = MagicMock()
        mock_inner.get.return_value = []
        analyser, _ = self._make_analyser(mock_inner)
        result = analyser.get_many_faces(MagicMock())
        assert result == []


# ---------------------------------------------------------------------------
# FaceAnalyser.set_det_size
# ---------------------------------------------------------------------------


class TestFaceAnalyserSetDetSize:
    def _make_analyser(self):
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        mock_inner = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=mock_inner):
            analyser = FaceAnalyser(cfg)
        return analyser

    def test_no_op_when_size_unchanged(self):
        analyser = self._make_analyser()
        with patch("modules.face_analyser._FaceAnalysis") as MockFA:
            analyser.set_det_size((320, 320))  # same as default
        MockFA.assert_not_called()

    def test_recreates_analyser_on_size_change(self):
        analyser = self._make_analyser()
        new_mock = MagicMock()
        with patch("modules.face_analyser._FaceAnalysis", return_value=new_mock) as MockFA:
            analyser.set_det_size((160, 160))
        MockFA.assert_called_once()
        new_mock.prepare.assert_called_once_with(ctx_id=0, det_size=(160, 160))

    def test_thread_safe_set_det_size(self):
        """Concurrent calls to set_det_size must not corrupt the instance."""
        from modules.face_analyser import FaceAnalyser, FaceAnalyserConfig

        cfg = FaceAnalyserConfig(providers=["CPUExecutionProvider"])
        call_count = [0]

        def counting_fa(*args, **kwargs):
            call_count[0] += 1
            m = MagicMock()
            return m

        with patch("modules.face_analyser._FaceAnalysis", side_effect=counting_fa):
            analyser = FaceAnalyser(cfg)

        sizes = [(160, 160), (320, 320)] * 10
        errors = []

        def flip(size):
            try:
                with patch("modules.face_analyser._FaceAnalysis", side_effect=counting_fa):
                    analyser.set_det_size(size)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=flip, args=(s,)) for s in sizes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread-safety errors: {errors}"

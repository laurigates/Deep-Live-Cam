"""Tests for injectable execution providers — Issue #62.

Verifies that get_face_swapper() and get_face_enhancer() accept an explicit
providers list instead of reading from modules.globals, so they can be tested
without populating global state.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# face_swapper injectable providers
# ---------------------------------------------------------------------------


class TestFaceSwapperInjectableProviders:
    def test_get_face_swapper_accepts_providers_parameter(self):
        """get_face_swapper() must accept an optional providers keyword argument."""
        import inspect

        from modules.processors.frame import face_swapper

        sig = inspect.signature(face_swapper.get_face_swapper)
        assert "providers" in sig.parameters

    def test_get_face_swapper_uses_injected_providers_not_globals(self):
        """When providers is passed, globals.execution_providers must not be read."""
        from modules.processors.frame import face_swapper

        injected = ["CPUExecutionProvider"]
        captured_providers = []

        def fake_build_providers_config(providers):
            captured_providers.extend(providers)
            return providers

        fake_model = MagicMock()
        with (
            patch(
                "modules.processors.frame.face_swapper.build_providers_config", side_effect=fake_build_providers_config
            ),
            patch("insightface.model_zoo.get_model", return_value=fake_model),
            patch("modules.processors.frame.face_swapper.IS_APPLE_SILICON", False),
        ):
            # Reset singleton so initialization runs
            original = face_swapper.FACE_SWAPPER
            face_swapper.FACE_SWAPPER = None
            try:
                face_swapper.get_face_swapper(providers=injected)
            finally:
                face_swapper.FACE_SWAPPER = original

        assert captured_providers == injected, f"Expected {injected}, got {captured_providers}"

    def test_get_face_swapper_falls_back_to_globals_when_no_providers(self):
        """When providers is not passed, globals.execution_providers is used."""
        import modules.globals
        from modules.processors.frame import face_swapper

        modules.globals.execution_providers = ["CPUExecutionProvider"]
        captured_providers = []

        def fake_build_providers_config(providers):
            captured_providers.extend(providers)
            return providers

        fake_model = MagicMock()
        with (
            patch(
                "modules.processors.frame.face_swapper.build_providers_config", side_effect=fake_build_providers_config
            ),
            patch("insightface.model_zoo.get_model", return_value=fake_model),
            patch("modules.processors.frame.face_swapper.IS_APPLE_SILICON", False),
        ):
            original = face_swapper.FACE_SWAPPER
            face_swapper.FACE_SWAPPER = None
            try:
                face_swapper.get_face_swapper()
            finally:
                face_swapper.FACE_SWAPPER = original

        assert captured_providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# face_enhancer injectable providers
# ---------------------------------------------------------------------------


class TestFaceEnhancerInjectableProviders:
    def test_get_face_enhancer_accepts_providers_parameter(self):
        """get_face_enhancer() must accept an optional providers keyword argument."""
        import inspect

        from modules.processors.frame import face_enhancer

        sig = inspect.signature(face_enhancer.get_face_enhancer)
        assert "providers" in sig.parameters

    def test_get_face_enhancer_uses_injected_providers_not_globals(self):
        """When providers is passed, that list is forwarded to InferenceSession."""
        from modules.processors.frame import face_enhancer

        injected = ["CPUExecutionProvider"]
        captured_providers = []

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name="input", shape=[1], type="float")]
        fake_session.get_outputs.return_value = [MagicMock(name="out", shape=[1], type="float")]
        fake_session.get_providers.return_value = injected

        def fake_inference_session(model_path, sess_options=None, providers=None, **kwargs):
            if providers:
                captured_providers.extend(providers)
            return fake_session

        with patch("os.path.exists", return_value=True):
            with patch(
                "modules.processors.frame.face_enhancer.onnxruntime.InferenceSession",
                side_effect=fake_inference_session,
            ):
                original = face_enhancer.FACE_ENHANCER
                face_enhancer.FACE_ENHANCER = None
                try:
                    face_enhancer.get_face_enhancer(providers=injected)
                finally:
                    face_enhancer.FACE_ENHANCER = original

        assert captured_providers == injected

    def test_get_face_enhancer_falls_back_to_globals_when_no_providers(self):
        """When providers is not passed, globals.execution_providers is used."""
        import modules.globals
        from modules.processors.frame import face_enhancer

        modules.globals.execution_providers = ["CPUExecutionProvider"]
        captured_providers = []

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [MagicMock(name="input", shape=[1], type="float")]
        fake_session.get_outputs.return_value = [MagicMock(name="out", shape=[1], type="float")]
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]

        def fake_inference_session(model_path, sess_options=None, providers=None, **kwargs):
            if providers:
                captured_providers.extend(providers)
            return fake_session

        with patch("os.path.exists", return_value=True):
            with patch(
                "modules.processors.frame.face_enhancer.onnxruntime.InferenceSession",
                side_effect=fake_inference_session,
            ):
                original = face_enhancer.FACE_ENHANCER
                face_enhancer.FACE_ENHANCER = None
                try:
                    face_enhancer.get_face_enhancer()
                finally:
                    face_enhancer.FACE_ENHANCER = original

        assert captured_providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# face_analyser already injectable — verify consistent API
# ---------------------------------------------------------------------------


class TestFaceAnalyserAlreadyInjectable:
    def test_get_face_analyser_accepts_config_parameter(self):
        """Verify face_analyser already follows the injectable pattern."""
        import inspect

        from modules import face_analyser

        sig = inspect.signature(face_analyser.get_face_analyser)
        assert "config" in sig.parameters

    def test_all_three_processors_have_injectable_providers(self):
        """Confirm all three processor loader functions accept injection."""
        import inspect

        from modules import face_analyser
        from modules.processors.frame import face_enhancer, face_swapper

        assert "config" in inspect.signature(face_analyser.get_face_analyser).parameters
        assert "providers" in inspect.signature(face_swapper.get_face_swapper).parameters
        assert "providers" in inspect.signature(face_enhancer.get_face_enhancer).parameters

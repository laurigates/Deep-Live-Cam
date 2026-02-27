"""Tests for refactoring face_analyser to accept injectable ProcessingConfig.

This file has two sections:

Section 1 — Config structure specs (verify ProcessingConfig has the right fields).
These tests were written first (TDD RED) and already pass once ProcessingConfig exists.

Section 2 — Behavioral tests for the injectable API (verify functions actually use
the injected config instead of reading from modules.globals).
"""
import pytest
import threading
from unittest.mock import MagicMock, patch
from modules.processing_config import ProcessingConfig


def test_face_analyser_can_accept_execution_providers_from_config():
    """
    Demonstrate that face_analyser initialization should accept
    execution_providers from ProcessingConfig.

    Currently, get_face_analyser() reads modules.globals.execution_providers.
    After refactoring, it should accept a config parameter.
    """
    from modules.processing_config_factory import build_config_from_globals

    config = build_config_from_globals()

    # The config contains the providers
    assert hasattr(config, 'execution_providers')
    assert isinstance(config.execution_providers, list)

    # These could be passed to face_analyser instead of reading globals
    # Example of future API:
    #   get_face_analyser(config=config)
    #   set_det_size(config, new_size)


def test_config_contains_face_analyser_configuration_fields():
    """
    Verify ProcessingConfig has all configuration fields face_analyser needs.

    Note: Runtime data like source_target_map and simple_map are excluded,
    as these are mutable state that grows during processing, not configuration.
    """
    from modules.processing_config_factory import build_config_from_globals

    config = build_config_from_globals()

    # Configuration fields face_analyser needs:
    assert hasattr(config, 'execution_providers')
    assert hasattr(config, 'many_faces')
    assert hasattr(config, 'target_path')
    assert hasattr(config, 'map_lock')

    # All configuration fields present and valid
    assert config.execution_providers is not None
    assert isinstance(config.many_faces, bool)
    assert config.map_lock is not None


def test_config_face_detection_settings():
    """Verify config has face detection configuration fields."""
    from modules.processing_config_factory import build_config_from_globals

    config = build_config_from_globals()

    # Face detection configuration
    assert hasattr(config, 'face_confidence_threshold')
    assert hasattr(config, 'detection_interval')
    assert hasattr(config, 'detection_cache_size')
    assert hasattr(config, 'face_analyser_det_size')

    # Values are valid
    assert 0 <= config.face_confidence_threshold <= 1
    assert config.detection_interval > 0
    assert config.detection_cache_size > 0
    assert isinstance(config.face_analyser_det_size, tuple)
    assert len(config.face_analyser_det_size) == 2


# ---------------------------------------------------------------------------
# Section 2 — Behavioral tests for the injectable API
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_face_analyser_singleton():
    """Reset the FACE_ANALYSER singleton before each test to ensure isolation."""
    import modules.face_analyser as fa
    original = fa.FACE_ANALYSER
    fa.FACE_ANALYSER = None
    yield
    fa.FACE_ANALYSER = original


def test_get_face_analyser_uses_execution_providers_from_config():
    """get_face_analyser(config=config) must use config.execution_providers, not globals."""
    import modules.face_analyser as fa

    config = ProcessingConfig(execution_providers=['CPUExecutionProvider'])

    mock_instance = MagicMock()
    with patch('modules.face_analyser._FaceAnalysis', return_value=mock_instance) as MockFA:
        result = fa.get_face_analyser(config=config)

    # FaceAnalysis was created with CPU provider (CoreML is filtered but CPU stays)
    MockFA.assert_called_once()
    _, call_kwargs = MockFA.call_args
    providers_used = call_kwargs['providers']
    assert 'CPUExecutionProvider' in providers_used

    # prepare() was called on the instance
    mock_instance.prepare.assert_called_once()

    # returned object is the mock instance
    assert result is mock_instance


def test_detect_faces_uses_many_faces_from_config():
    """detect_faces(frame, config=config) dispatches based on config.many_faces."""
    import modules.face_analyser as fa

    frame = MagicMock()
    mock_face = MagicMock()

    with patch.object(fa, 'get_many_faces', return_value=[mock_face]) as mock_many, \
         patch.object(fa, 'get_one_face', return_value=mock_face) as mock_one:

        # many_faces=True → get_many_faces
        config_many = ProcessingConfig(many_faces=True)
        result_many = fa.detect_faces(frame, config=config_many)
        mock_many.assert_called_once_with(frame)
        mock_one.assert_not_called()
        assert result_many == [mock_face]

        mock_many.reset_mock()
        mock_one.reset_mock()

        # many_faces=False → get_one_face
        config_one = ProcessingConfig(many_faces=False)
        result_one = fa.detect_faces(frame, config=config_one)
        mock_one.assert_called_once_with(frame)
        mock_many.assert_not_called()
        assert result_one == [mock_face]


def test_set_det_size_uses_config_providers():
    """set_det_size(size, config=config) must recreate FaceAnalysis with config providers."""
    import modules.face_analyser as fa

    # Force a different det_size so the early-return guard doesn't fire
    fa._CURRENT_DET_SIZE = (320, 320)

    config = ProcessingConfig(execution_providers=['CPUExecutionProvider'])
    new_size = (160, 160)

    mock_instance = MagicMock()
    with patch('modules.face_analyser._FaceAnalysis', return_value=mock_instance) as MockFA:
        fa.set_det_size(new_size, config=config)

    MockFA.assert_called_once()
    _, call_kwargs = MockFA.call_args
    providers_used = call_kwargs['providers']
    assert 'CPUExecutionProvider' in providers_used
    mock_instance.prepare.assert_called_once_with(ctx_id=0, det_size=new_size)

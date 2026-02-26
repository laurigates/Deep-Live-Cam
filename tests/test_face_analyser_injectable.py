"""Tests for refactoring face_analyser to accept injectable ProcessingConfig.

This test suite documents the desired API for an injectable face_analyser,
showing how key functions should accept ProcessingConfig instead of reading
from modules.globals.

Full refactoring is staged across multiple PRs to minimize risk.

Note: Some globals (source_target_map, simple_map) are runtime data, not configuration.
These may remain as globals or be passed through different mechanisms.
"""
import pytest
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

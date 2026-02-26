"""Tests for processing config factory functions."""
import pytest
from unittest.mock import patch, MagicMock
from modules.processing_config_factory import build_config_from_globals
from modules.processing_config import ProcessingConfig


def test_build_config_from_globals_creates_config():
    """Verify that build_config_from_globals() creates a complete ProcessingConfig."""
    config = build_config_from_globals()

    assert isinstance(config, ProcessingConfig)
    assert config.execution_providers is not None
    assert isinstance(config.frame_processors, list)


def test_build_config_from_globals_copies_mutable_fields():
    """
    Ensure that mutable fields (lists, dicts) are copied, not referenced.

    This prevents external mutations to the original globals from affecting
    the config, and vice versa.
    """
    import modules.globals

    original_providers = modules.globals.execution_providers.copy()
    original_processors = modules.globals.frame_processors.copy()

    config = build_config_from_globals()

    # Modify the config's lists
    config.execution_providers.append('NewProvider')
    config.frame_processors.append('new_processor')

    # Verify globals are unchanged
    assert modules.globals.execution_providers == original_providers
    assert modules.globals.frame_processors == original_processors


def test_build_config_from_globals_preserves_values():
    """Verify that all global values are properly transferred to config."""
    import modules.globals

    # Set some known values
    modules.globals.many_faces = True
    modules.globals.mouth_mask = True
    modules.globals.opacity = 0.7
    modules.globals.live_max_fps = 45
    modules.globals.keyframe_interval = 3

    config = build_config_from_globals()

    # Verify values match
    assert config.many_faces is True
    assert config.mouth_mask is True
    assert config.opacity == 0.7
    assert config.live_max_fps == 45
    assert config.keyframe_interval == 3


def test_build_config_from_globals_preserves_map_lock():
    """Verify that the synchronization lock is preserved."""
    import modules.globals
    import threading

    config = build_config_from_globals()

    # The config should reference the same lock object
    assert config.map_lock is modules.globals.MAP_LOCK
    # Verify it has the lock interface (acquire/release methods)
    assert hasattr(config.map_lock, 'acquire')
    assert hasattr(config.map_lock, 'release')


def test_config_fields_not_mutated_by_global_changes():
    """
    Verify that changes to globals don't affect an already-built config.

    This tests that config is independent of future global mutations.
    """
    import modules.globals

    # Build config
    config = build_config_from_globals()
    original_value = config.opacity

    # Modify global
    modules.globals.opacity = 0.5

    # Config should still have the original value
    assert config.opacity == original_value
    assert modules.globals.opacity == 0.5

    # Reset for next test
    modules.globals.opacity = original_value


def test_build_config_from_globals_handles_none_values():
    """Verify that None values in globals are properly transferred."""
    import modules.globals

    modules.globals.source_path = None
    modules.globals.target_path = None
    modules.globals.execution_threads = None

    config = build_config_from_globals()

    assert config.source_path is None
    assert config.target_path is None
    assert config.execution_threads is None

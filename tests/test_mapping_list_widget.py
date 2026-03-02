"""Tests for MappingListWidget (Phase 2 — Unified Face Mapping UI).

These tests verify the widget logic without requiring a live Tkinter display.
We mock CTk widgets so tests run in CI without a GUI.
"""
from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

from modules.mapping_list import MappingList, MappingEntry


# ---------------------------------------------------------------------------
# We cannot import the widget module at module level because it imports CTk.
# Instead, we patch customtkinter before importing.
# ---------------------------------------------------------------------------

def _import_widget_module():
    """Import ui_mapping_list with CTk mocked out for headless testing."""
    # The module imports customtkinter at the top level; we need it available
    # but don't need a real Tk instance.
    import modules.ui_mapping_list as mod
    return mod


class TestMappingListWidgetCreation:
    """Test that the widget initialises from MappingList state."""

    @patch("modules.ui_mapping_list.ctk")
    def test_widget_constructs_without_error(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        assert widget is not None

    @patch("modules.ui_mapping_list.ctk")
    def test_widget_subscribes_to_changes(self, mock_ctk):
        """Widget registers an on_change callback during construction."""
        mod = _import_widget_module()
        ml = MappingList()
        parent = MagicMock()
        initial_cb_count = len(ml._on_change)
        mod.MappingListWidget(parent, ml)
        assert len(ml._on_change) == initial_cb_count + 1

    @patch("modules.ui_mapping_list.ctk")
    def test_remove_button_hidden_for_single_entry(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        # Single entry — remove button should not be shown
        assert widget._should_show_remove() is False

    @patch("modules.ui_mapping_list.ctk")
    def test_remove_button_shown_for_multiple_entries(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        ml.add_entry()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        assert widget._should_show_remove() is True

    @patch("modules.ui_mapping_list.ctk")
    def test_thumb_size_single_entry(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        assert widget._thumb_size() == (120, 120)

    @patch("modules.ui_mapping_list.ctk")
    def test_thumb_size_multiple_entries(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        ml.add_entry()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        assert widget._thumb_size() == (80, 80)


class TestMappingListWidgetCallbacks:
    """Test that widget button handlers call the right MappingList methods."""

    @patch("modules.ui_mapping_list.ctk")
    def test_on_add_calls_add_entry(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        initial_count = len(ml.get_entries())
        widget._on_add()
        assert len(ml.get_entries()) == initial_count + 1

    @patch("modules.ui_mapping_list.ctk")
    def test_on_remove_calls_remove_entry(self, mock_ctk):
        mod = _import_widget_module()
        ml = MappingList()
        ml.add_entry()
        parent = MagicMock()
        widget = mod.MappingListWidget(parent, ml)
        widget._on_remove(0)
        entries = ml.get_entries()
        assert len(entries) == 1
        assert entries[0].id == 1

"""Tests for the MappingList data model (Phase 1 — Unified Face Mapping UI)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from modules.mapping_list import MappingEntry, MappingList


# ---------------------------------------------------------------------------
# MappingEntry basics
# ---------------------------------------------------------------------------

class TestMappingEntry:
    def test_default_fields(self):
        entry = MappingEntry(id=0)
        assert entry.id == 0
        assert entry.source_path is None
        assert entry.source_face is None
        assert entry.source_cv2 is None
        assert entry.pin_path is None
        assert entry.pin_face is None
        assert entry.pin_cv2 is None

    def test_with_source(self):
        face = MagicMock()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        entry = MappingEntry(id=1, source_path="/a.png", source_face=face, source_cv2=img)
        assert entry.source_path == "/a.png"
        assert entry.source_face is face
        assert entry.source_cv2 is img

    def test_with_pin(self):
        face = MagicMock()
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        entry = MappingEntry(id=2, pin_path="/b.png", pin_face=face, pin_cv2=img)
        assert entry.pin_path == "/b.png"
        assert entry.pin_face is face


# ---------------------------------------------------------------------------
# MappingList — add / remove / get
# ---------------------------------------------------------------------------

class TestMappingListBasics:
    def test_starts_with_one_entry(self):
        ml = MappingList()
        assert len(ml.get_entries()) == 1
        assert ml.get_entries()[0].id == 0

    def test_add_entry(self):
        ml = MappingList()
        entry = ml.add_entry()
        assert entry.id == 1
        assert len(ml.get_entries()) == 2

    def test_add_multiple_entries(self):
        ml = MappingList()
        ml.add_entry()
        ml.add_entry()
        entries = ml.get_entries()
        assert len(entries) == 3
        assert [e.id for e in entries] == [0, 1, 2]

    def test_remove_entry(self):
        ml = MappingList()
        ml.add_entry()
        ml.remove_entry(0)
        entries = ml.get_entries()
        assert len(entries) == 1
        assert entries[0].id == 1

    def test_remove_nonexistent_entry(self):
        ml = MappingList()
        ml.remove_entry(999)  # should not raise
        assert len(ml.get_entries()) == 1

    def test_remove_last_entry_leaves_empty(self):
        ml = MappingList()
        ml.remove_entry(0)
        assert len(ml.get_entries()) == 0

    def test_clear_all(self):
        ml = MappingList()
        ml.add_entry()
        ml.add_entry()
        ml.clear_all()
        assert len(ml.get_entries()) == 0

    def test_get_entries_returns_snapshot(self):
        ml = MappingList()
        entries = ml.get_entries()
        entries.append(MappingEntry(id=99))
        assert len(ml.get_entries()) == 1  # original unaffected


# ---------------------------------------------------------------------------
# MappingList — set_source / set_pin / clear_pin
# ---------------------------------------------------------------------------

class TestMappingListSourcePin:
    def test_set_source(self):
        ml = MappingList()
        face = MagicMock()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        ml.set_source(0, "/src.png", img, face)
        entry = ml.get_entries()[0]
        assert entry.source_path == "/src.png"
        assert entry.source_face is face
        assert entry.source_cv2 is img

    def test_set_source_nonexistent_id(self):
        ml = MappingList()
        ml.set_source(999, "/x.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        # no crash, entry 0 unchanged
        assert ml.get_entries()[0].source_path is None

    def test_set_pin(self):
        ml = MappingList()
        face = MagicMock()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        ml.set_pin(0, "/pin.png", img, face)
        entry = ml.get_entries()[0]
        assert entry.pin_path == "/pin.png"
        assert entry.pin_face is face
        assert entry.pin_cv2 is img

    def test_clear_pin(self):
        ml = MappingList()
        ml.set_pin(0, "/pin.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        ml.clear_pin(0)
        entry = ml.get_entries()[0]
        assert entry.pin_path is None
        assert entry.pin_face is None
        assert entry.pin_cv2 is None


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------

class TestDerivedState:
    def test_effective_map_faces_single_no_pin(self):
        ml = MappingList()
        assert ml.effective_map_faces() is False

    def test_effective_map_faces_multiple_entries(self):
        ml = MappingList()
        ml.add_entry()
        assert ml.effective_map_faces() is True

    def test_effective_map_faces_with_pin(self):
        ml = MappingList()
        ml.set_pin(0, "/pin.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        assert ml.effective_map_faces() is True

    def test_effective_source_path_none(self):
        ml = MappingList()
        assert ml.effective_source_path() is None

    def test_effective_source_path_set(self):
        ml = MappingList()
        ml.set_source(0, "/src.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        assert ml.effective_source_path() == "/src.png"

    def test_source_count_zero(self):
        ml = MappingList()
        assert ml.source_count() == 0

    def test_source_count_with_sources(self):
        ml = MappingList()
        ml.set_source(0, "/a.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        ml.add_entry()
        assert ml.source_count() == 1
        ml.set_source(1, "/b.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        assert ml.source_count() == 2


# ---------------------------------------------------------------------------
# On-change callbacks
# ---------------------------------------------------------------------------

class TestOnChangeCallbacks:
    def test_add_triggers_callback(self):
        cb = MagicMock()
        ml = MappingList()
        ml.on_change(cb)
        ml.add_entry()
        cb.assert_called_once()

    def test_remove_triggers_callback(self):
        cb = MagicMock()
        ml = MappingList()
        ml.on_change(cb)
        ml.remove_entry(0)
        cb.assert_called_once()

    def test_set_source_triggers_callback(self):
        cb = MagicMock()
        ml = MappingList()
        ml.on_change(cb)
        ml.set_source(0, "/s.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        cb.assert_called_once()

    def test_set_pin_triggers_callback(self):
        cb = MagicMock()
        ml = MappingList()
        ml.on_change(cb)
        ml.set_pin(0, "/p.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        cb.assert_called_once()

    def test_clear_all_triggers_callback(self):
        cb = MagicMock()
        ml = MappingList()
        ml.on_change(cb)
        ml.clear_all()
        cb.assert_called_once()


# ---------------------------------------------------------------------------
# Sync to FaceMapStore
# ---------------------------------------------------------------------------

class TestSyncToStore:
    def test_sync_single_source_no_pin(self):
        """Single source with no pin should set entries but not simplify."""
        from modules.face_map_store import FaceMapStore
        store = FaceMapStore()
        ml = MappingList()
        face = MagicMock()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        ml.set_source(0, "/src.png", img, face)
        ml.sync_to_store(store)
        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0]['source']['face'] is face
        assert entries[0]['source']['cv2'] is img

    def test_sync_with_pin_calls_simplify(self):
        """Entries with pins should have both source and target, and simplify is called."""
        from modules.face_map_store import FaceMapStore
        store = FaceMapStore()
        ml = MappingList()
        src_face = MagicMock()
        src_face.normed_embedding = np.ones(512)
        pin_face = MagicMock()
        pin_face.normed_embedding = np.ones(512)
        src_img = np.zeros((50, 50, 3), dtype=np.uint8)
        pin_img = np.zeros((50, 50, 3), dtype=np.uint8)
        ml.set_source(0, "/src.png", src_img, src_face)
        ml.set_pin(0, "/pin.png", pin_img, pin_face)
        ml.sync_to_store(store)
        entries = store.get_entries()
        assert len(entries) == 1
        assert 'source' in entries[0]
        assert 'target' in entries[0]
        # simplify should have been called — check simple map
        simple = store.get_simple_map()
        assert len(simple.get('source_faces', [])) == 1

    def test_sync_skips_entries_without_source(self):
        from modules.face_map_store import FaceMapStore
        store = FaceMapStore()
        ml = MappingList()
        ml.add_entry()  # entry 1 — no source
        ml.sync_to_store(store)
        assert len(store.get_entries()) == 0

    def test_sync_multiple_sources(self):
        from modules.face_map_store import FaceMapStore
        store = FaceMapStore()
        ml = MappingList()
        for i in range(2):
            if i > 0:
                ml.add_entry()
            face = MagicMock()
            face.normed_embedding = np.ones(512)
            ml.set_source(i, f"/{i}.png", np.zeros((10, 10, 3), dtype=np.uint8), face)
        ml.sync_to_store(store)
        assert len(store.get_entries()) == 2


# ---------------------------------------------------------------------------
# Persistence (to_dict / restore_from_dict)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_round_trip_empty(self):
        ml = MappingList()
        data = ml.to_dict()
        ml2 = MappingList()
        ml2.clear_all()
        ml2.restore_from_dict(data)
        assert len(ml2.get_entries()) == 1

    def test_round_trip_with_source(self):
        ml = MappingList()
        ml.set_source(0, "/src.png", np.zeros((10, 10, 3), dtype=np.uint8), MagicMock())
        data = ml.to_dict()
        ml2 = MappingList()
        ml2.clear_all()
        ml2.restore_from_dict(data)
        entries = ml2.get_entries()
        assert len(entries) == 1
        assert entries[0].source_path == "/src.png"
        # face objects are not serialized
        assert entries[0].source_face is None

    def test_round_trip_with_pin(self):
        ml = MappingList()
        ml.set_pin(0, "/pin.png", np.zeros((10, 10, 3), dtype=np.uint8), MagicMock())
        data = ml.to_dict()
        ml2 = MappingList()
        ml2.clear_all()
        ml2.restore_from_dict(data)
        entries = ml2.get_entries()
        assert entries[0].pin_path == "/pin.png"
        assert entries[0].pin_face is None

    def test_round_trip_multiple_entries(self):
        ml = MappingList()
        ml.set_source(0, "/a.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        ml.add_entry()
        ml.set_source(1, "/b.png", np.zeros((1, 1, 3), dtype=np.uint8), MagicMock())
        data = ml.to_dict()
        ml2 = MappingList()
        ml2.clear_all()
        ml2.restore_from_dict(data)
        entries = ml2.get_entries()
        assert len(entries) == 2
        assert entries[0].source_path == "/a.png"
        assert entries[1].source_path == "/b.png"

    def test_to_dict_excludes_cv2_and_face(self):
        ml = MappingList()
        ml.set_source(0, "/src.png", np.zeros((10, 10, 3), dtype=np.uint8), MagicMock())
        data = ml.to_dict()
        entry = data[0]
        assert "source_cv2" not in entry
        assert "source_face" not in entry
        assert "pin_cv2" not in entry
        assert "pin_face" not in entry

    def test_restore_from_empty_list(self):
        ml = MappingList()
        ml.restore_from_dict([])
        # Should have no entries
        assert len(ml.get_entries()) == 0

    def test_migrate_from_source_path(self):
        """Old state file has source_path but no mappings key — should create single mapping."""
        ml = MappingList()
        ml.clear_all()
        ml.restore_from_source_path("/legacy.png")
        entries = ml.get_entries()
        assert len(entries) == 1
        assert entries[0].source_path == "/legacy.png"

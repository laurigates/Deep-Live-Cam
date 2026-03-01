"""Tests for FaceMapStore — thread-safe face mapping state (Issue #59)."""

import threading
from unittest.mock import MagicMock

import pytest

from modules.face_map_store import FaceMapStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_face(embedding=None):
    """Create a minimal mock face object."""
    face = MagicMock()
    face.normed_embedding = embedding if embedding is not None else [0.1, 0.2]
    face.bbox = [10.0, 20.0, 110.0, 120.0]
    return face


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_starts_empty(self):
        store = FaceMapStore()
        assert store.get_entries() == []

    def test_simple_map_starts_empty(self):
        store = FaceMapStore()
        assert store.get_simple_map() == {}


# ---------------------------------------------------------------------------
# add_blank
# ---------------------------------------------------------------------------


class TestAddBlank:
    def test_adds_entry_with_id_zero_when_empty(self):
        store = FaceMapStore()
        store.add_blank()
        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0]["id"] == 0

    def test_subsequent_blank_ids_increment(self):
        store = FaceMapStore()
        store.add_blank()
        store.add_blank()
        entries = store.get_entries()
        assert [e["id"] for e in entries] == [0, 1]

    def test_blank_entry_has_no_source_or_target(self):
        store = FaceMapStore()
        store.add_blank()
        entry = store.get_entries()[0]
        assert "source" not in entry
        assert "target" not in entry


# ---------------------------------------------------------------------------
# set_entries / get_entries
# ---------------------------------------------------------------------------


class TestSetEntries:
    def test_replaces_all_entries(self):
        store = FaceMapStore()
        store.add_blank()
        new_entries = [{"id": 99, "source": {}, "target": {}}]
        store.set_entries(new_entries)
        assert store.get_entries() == new_entries

    def test_get_entries_returns_snapshot(self):
        store = FaceMapStore()
        store.add_blank()
        snapshot = store.get_entries()
        # Mutating the snapshot must not affect the store
        snapshot.clear()
        assert len(store.get_entries()) == 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clears_entries(self):
        store = FaceMapStore()
        store.add_blank()
        store.clear()
        assert store.get_entries() == []

    def test_clears_simple_map(self):
        store = FaceMapStore()
        store.set_simple_map(["face"], ["emb"])
        store.clear()
        assert store.get_simple_map() == {}


# ---------------------------------------------------------------------------
# has_valid_map
# ---------------------------------------------------------------------------


class TestHasValidMap:
    def test_returns_false_when_empty(self):
        store = FaceMapStore()
        assert store.has_valid_map() is False

    def test_returns_false_when_only_blank_entries(self):
        store = FaceMapStore()
        store.add_blank()
        assert store.has_valid_map() is False

    def test_returns_false_when_entry_missing_target(self):
        store = FaceMapStore()
        store.set_entries([{"id": 0, "source": {"face": make_face()}}])
        assert store.has_valid_map() is False

    def test_returns_true_when_entry_has_source_and_target(self):
        store = FaceMapStore()
        store.set_entries([{"id": 0, "source": {"face": make_face()}, "target": {"face": make_face()}}])
        assert store.has_valid_map() is True


# ---------------------------------------------------------------------------
# default_source_face
# ---------------------------------------------------------------------------


class TestDefaultSourceFace:
    def test_returns_none_when_no_source(self):
        store = FaceMapStore()
        assert store.default_source_face() is None

    def test_returns_first_source_face(self):
        face_a = make_face()
        face_b = make_face()
        store = FaceMapStore()
        store.set_entries(
            [
                {"id": 0, "source": {"face": face_a}},
                {"id": 1, "source": {"face": face_b}},
            ]
        )
        assert store.default_source_face() is face_a


# ---------------------------------------------------------------------------
# simple_map
# ---------------------------------------------------------------------------


class TestSimpleMap:
    def test_set_and_get_simple_map(self):
        store = FaceMapStore()
        faces = [make_face()]
        embeddings = [[0.1, 0.2]]
        store.set_simple_map(faces, embeddings)
        result = store.get_simple_map()
        assert result["source_faces"] is faces
        assert result["target_embeddings"] is embeddings

    def test_get_simple_map_returns_copy(self):
        store = FaceMapStore()
        store.set_simple_map([], [])
        snapshot = store.get_simple_map()
        snapshot["extra"] = "injected"
        assert "extra" not in store.get_simple_map()

    def test_set_simple_map_from_entries(self):
        """Verify simplify() builds simple map from paired entries."""
        face_s = make_face([0.5, 0.6])
        face_t = make_face([0.7, 0.8])
        store = FaceMapStore()
        store.set_entries([{"id": 0, "source": {"face": face_s}, "target": {"face": face_t}}])
        store.simplify()
        simple = store.get_simple_map()
        assert simple["source_faces"] == [face_s]
        assert simple["target_embeddings"] == [face_t.normed_embedding]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_blank_produces_unique_ids(self):
        store = FaceMapStore()
        threads = [threading.Thread(target=store.add_blank) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entries = store.get_entries()
        ids = [e["id"] for e in entries]
        assert len(ids) == 50
        assert len(set(ids)) == 50, "Duplicate IDs detected — lock not held during add_blank"

    def test_concurrent_set_and_get_entries_does_not_raise(self):
        store = FaceMapStore()
        errors = []

        def writer():
            try:
                for i in range(100):
                    store.set_entries([{"id": i}])
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(100):
                    store.get_entries()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_store_singleton_is_face_map_store_instance(self):
        from modules.face_map_store import STORE

        assert isinstance(STORE, FaceMapStore)

"""Tests for modules/face_analyser.py — unit-testable behaviour."""

from modules.face_map_store import STORE as _MAP_STORE


def test_default_target_face_empty_map():
    """default_target_face must not crash when the map is empty."""
    _MAP_STORE.clear()
    from modules import face_analyser

    face_analyser.default_target_face()


def test_default_target_face_no_faces_in_frame():
    """default_target_face must not crash when no frame has faces (best_face stays None)."""
    fake_map = [{"target_faces_in_frame": [{"faces": [], "location": "/tmp/fake.png"}]}]
    _MAP_STORE.set_entries(fake_map)
    try:
        from modules import face_analyser

        face_analyser.default_target_face()
    finally:
        _MAP_STORE.clear()

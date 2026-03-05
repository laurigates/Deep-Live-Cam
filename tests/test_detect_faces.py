"""Tests for detect_faces in modules/face_analyser.py (Phase 5)."""

from unittest.mock import MagicMock, patch

import numpy as np


@patch("modules.face_analyser.modules.globals")
@patch("modules.face_analyser.get_many_faces")
def test_many_faces_mode_returns_all(mock_get_many, mock_globals):
    from modules.face_analyser import detect_faces

    mock_globals.many_faces = True
    fake_faces = [MagicMock(), MagicMock()]
    mock_get_many.return_value = fake_faces

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert result == fake_faces


@patch("modules.face_analyser.modules.globals")
@patch("modules.face_analyser.get_one_face")
def test_single_face_mode_returns_list_of_one(mock_get_one, mock_globals):
    from modules.face_analyser import detect_faces

    mock_globals.many_faces = False
    fake_face = MagicMock()
    mock_get_one.return_value = fake_face

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert result == [fake_face]


@patch("modules.face_analyser.modules.globals")
@patch("modules.face_analyser.get_one_face")
def test_no_face_returns_empty_list_single_mode(mock_get_one, mock_globals):
    from modules.face_analyser import detect_faces

    mock_globals.many_faces = False
    mock_get_one.return_value = None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert result == []


@patch("modules.face_analyser.modules.globals")
@patch("modules.face_analyser.get_many_faces")
def test_no_faces_returns_empty_list_many_mode(mock_get_many, mock_globals):
    from modules.face_analyser import detect_faces

    mock_globals.many_faces = True
    mock_get_many.return_value = None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert result == []


@patch("modules.face_analyser.modules.globals")
@patch("modules.face_analyser.get_many_faces")
def test_empty_faces_returns_empty_list(mock_get_many, mock_globals):
    from modules.face_analyser import detect_faces

    mock_globals.many_faces = True
    mock_get_many.return_value = []

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_faces(frame)
    assert result == []

"""Tests for batch multi-face swap (issue #48).

Verifies that batch_swap_faces() batches inference correctly, falls back
to sequential on error, and that process_frame/process_frame_v2 route
to batch vs. single-face paths appropriately.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers to build mock Face objects and sessions
# ---------------------------------------------------------------------------

def _make_face(embedding_dim=512, kps_shape=(5, 2)):
    """Create a minimal Face-like object with normed_embedding and kps."""
    face = SimpleNamespace()
    emb = np.random.randn(embedding_dim).astype(np.float32)
    face.normed_embedding = emb / np.linalg.norm(emb)
    face.kps = np.array([
        [40, 50], [88, 50], [64, 75], [45, 95], [83, 95]
    ], dtype=np.float32)
    face.bbox = np.array([20, 30, 108, 120], dtype=np.float32)
    return face


def _make_swapper(emap_shape=(512, 512), input_size=(128, 128)):
    """Create a mock face swapper with the attributes batch_swap_faces reads."""
    swapper = MagicMock()
    swapper.emap = np.random.randn(*emap_shape).astype(np.float32)
    swapper.input_size = input_size
    swapper.input_mean = 0.0
    swapper.input_std = 255.0
    swapper.input_names = ["target", "source"]
    swapper.output_names = ["output"]
    return swapper


def _make_session_run(n_faces, size=128):
    """Return a session.run side_effect that validates batch shapes and returns fake output."""
    def session_run(output_names, input_feed):
        blob = input_feed["target"]
        latent = input_feed["source"]
        assert blob.ndim == 4, f"Expected 4D blob, got {blob.ndim}D"
        assert blob.shape[0] == n_faces, f"Expected batch={n_faces}, got {blob.shape[0]}"
        assert blob.shape[1:] == (3, size, size)
        assert latent.shape[0] == n_faces
        # Return random predictions: (N, 3, 128, 128)
        pred = np.random.rand(n_faces, 3, size, size).astype(np.float32)
        return [pred]
    return session_run


# ---------------------------------------------------------------------------
# _paste_back
# ---------------------------------------------------------------------------

class TestPasteBack:
    def test_output_shape_matches_target(self):
        from modules.processors.frame.face_swapper import _paste_back
        bgr_fake = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)
        target_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = _paste_back(bgr_fake, aimg, M, target_img)
        assert result.shape == target_img.shape
        assert result.dtype == np.uint8

    def test_identity_transform_modifies_frame(self):
        from modules.processors.frame.face_swapper import _paste_back
        bgr_fake = np.full((128, 128, 3), 200, dtype=np.uint8)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)
        target_img = np.full((480, 640, 3), 50, dtype=np.uint8)
        result = _paste_back(bgr_fake, aimg, M, target_img)
        # The pasted region should differ from the original target
        assert not np.array_equal(result, target_img)


# ---------------------------------------------------------------------------
# batch_swap_faces
# ---------------------------------------------------------------------------

class TestBatchSwapFaces:

    @patch("modules.processors.frame.face_swapper.get_face_swapper")
    @patch("modules.processors.frame.face_swapper._apply_mouth_mask", side_effect=lambda f, *a: f)
    @patch("modules.processors.frame.face_swapper._apply_poisson_blend", side_effect=lambda f, *a: f)
    def test_batch_tensor_shapes_two_faces(self, _mock_blend, _mock_mouth, mock_get_swapper):
        """With 2 faces, session.run receives (2, 3, 128, 128) blob and (2, 512) latent."""
        from modules.processors.frame.face_swapper import batch_swap_faces
        import modules.globals
        modules.globals.mouth_mask = False
        modules.globals.poisson_blend = False
        modules.globals.opacity = 1.0

        swapper = _make_swapper()
        swapper.session.run.side_effect = _make_session_run(2)
        mock_get_swapper.return_value = swapper

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        sources = [_make_face(), _make_face()]
        targets = [_make_face(), _make_face()]

        result = batch_swap_faces(sources, targets, frame)

        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        swapper.session.run.assert_called_once()

    @patch("modules.processors.frame.face_swapper.get_face_swapper")
    @patch("modules.processors.frame.face_swapper._apply_mouth_mask", side_effect=lambda f, *a: f)
    @patch("modules.processors.frame.face_swapper._apply_poisson_blend", side_effect=lambda f, *a: f)
    def test_batch_three_faces(self, _mock_blend, _mock_mouth, mock_get_swapper):
        """With 3 faces, session.run receives batch dim = 3."""
        from modules.processors.frame.face_swapper import batch_swap_faces
        import modules.globals
        modules.globals.mouth_mask = False
        modules.globals.poisson_blend = False
        modules.globals.opacity = 1.0

        swapper = _make_swapper()
        swapper.session.run.side_effect = _make_session_run(3)
        mock_get_swapper.return_value = swapper

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        sources = [_make_face() for _ in range(3)]
        targets = [_make_face() for _ in range(3)]

        result = batch_swap_faces(sources, targets, frame)

        assert result.shape == frame.shape
        swapper.session.run.assert_called_once()

    @patch("modules.processors.frame.face_swapper.swap_face")
    @patch("modules.processors.frame.face_swapper.get_face_swapper")
    def test_fallback_on_session_error(self, mock_get_swapper, mock_swap_face):
        """If session.run raises, falls back to sequential swap_face calls."""
        import modules.processors.frame.face_swapper as mod
        from modules.processors.frame.face_swapper import batch_swap_faces
        import modules.globals
        modules.globals.mouth_mask = False
        modules.globals.poisson_blend = False
        modules.globals.opacity = 1.0

        # Reset the warning flag so the fallback path logs
        mod._batch_fallback_warned = False

        swapper = _make_swapper()
        swapper.session.run.side_effect = RuntimeError("Static shapes required")
        mock_get_swapper.return_value = swapper

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mock_swap_face.side_effect = lambda s, t, f: f  # pass-through

        sources = [_make_face(), _make_face()]
        targets = [_make_face(), _make_face()]

        result = batch_swap_faces(sources, targets, frame)

        assert result.shape == frame.shape
        assert mock_swap_face.call_count == 2

    @patch("modules.processors.frame.face_swapper.get_face_swapper")
    def test_returns_frame_when_no_valid_faces(self, mock_get_swapper):
        """If all faces are None, returns original frame unchanged."""
        from modules.processors.frame.face_swapper import batch_swap_faces

        swapper = _make_swapper()
        mock_get_swapper.return_value = swapper

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = batch_swap_faces([None, None], [None, None], frame)

        assert np.array_equal(result, frame)
        swapper.session.run.assert_not_called()

    @patch("modules.processors.frame.face_swapper.get_face_swapper")
    def test_returns_frame_when_swapper_none(self, mock_get_swapper):
        """If face swapper model is not loaded, returns original frame."""
        from modules.processors.frame.face_swapper import batch_swap_faces

        mock_get_swapper.return_value = None
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = batch_swap_faces([_make_face()], [_make_face()], frame)

        assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# process_frame routing
# ---------------------------------------------------------------------------

class TestProcessFrameRouting:

    @patch("modules.processors.frame.face_swapper.apply_post_processing", side_effect=lambda f, b: f)
    @patch("modules.processors.frame.face_swapper.batch_swap_faces")
    @patch("modules.processors.frame.face_swapper.get_many_faces")
    def test_many_faces_uses_batch_when_two_or_more(self, mock_get_many, mock_batch, _mock_post):
        """process_frame uses batch_swap_faces when many_faces and N >= 2."""
        from modules.processors.frame.face_swapper import process_frame
        import modules.globals
        modules.globals.many_faces = True
        modules.globals.opacity = 1.0

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        faces = [_make_face(), _make_face(), _make_face()]
        mock_get_many.return_value = faces
        mock_batch.return_value = frame.copy()

        process_frame(_make_face(), frame)

        mock_batch.assert_called_once()
        # Verify it was called with 3 source faces and 3 target faces
        args = mock_batch.call_args
        assert len(args[0][0]) == 3  # source_faces
        assert len(args[0][1]) == 3  # target_faces

    @patch("modules.processors.frame.face_swapper.apply_post_processing", side_effect=lambda f, b: f)
    @patch("modules.processors.frame.face_swapper.swap_face")
    @patch("modules.processors.frame.face_swapper.get_many_faces")
    def test_single_many_face_uses_swap_face(self, mock_get_many, mock_swap, _mock_post):
        """process_frame uses swap_face (not batch) when many_faces but only 1 detected."""
        from modules.processors.frame.face_swapper import process_frame
        import modules.globals
        modules.globals.many_faces = True
        modules.globals.opacity = 1.0

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        faces = [_make_face()]
        mock_get_many.return_value = faces
        mock_swap.return_value = frame.copy()

        process_frame(_make_face(), frame)

        mock_swap.assert_called_once()


# ---------------------------------------------------------------------------
# process_frame_v2 routing
# ---------------------------------------------------------------------------

class TestProcessFrameV2Routing:

    @patch("modules.processors.frame.face_swapper.apply_post_processing", side_effect=lambda f, b: f)
    @patch("modules.processors.frame.face_swapper.batch_swap_faces")
    @patch("modules.processors.frame.face_swapper._build_pairs_live")
    def test_v2_uses_batch_when_two_pairs(self, mock_build, mock_batch, _mock_post):
        """process_frame_v2 routes to batch when >= 2 valid pairs."""
        from modules.processors.frame.face_swapper import process_frame_v2
        import modules.globals
        modules.globals.opacity = 1.0
        modules.globals.target_path = None  # live mode

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        pairs = [(_make_face(), _make_face()), (_make_face(), _make_face())]
        mock_build.return_value = pairs
        mock_batch.return_value = frame.copy()

        process_frame_v2(frame)

        mock_batch.assert_called_once()

    @patch("modules.processors.frame.face_swapper.apply_post_processing", side_effect=lambda f, b: f)
    @patch("modules.processors.frame.face_swapper.swap_face")
    @patch("modules.processors.frame.face_swapper._build_pairs_live")
    def test_v2_uses_swap_face_for_single_pair(self, mock_build, mock_swap, _mock_post):
        """process_frame_v2 uses swap_face when only 1 valid pair."""
        from modules.processors.frame.face_swapper import process_frame_v2
        import modules.globals
        modules.globals.opacity = 1.0
        modules.globals.target_path = None  # live mode

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        pairs = [(_make_face(), _make_face())]
        mock_build.return_value = pairs
        mock_swap.return_value = frame.copy()

        process_frame_v2(frame)

        mock_swap.assert_called_once()

    @patch("modules.processors.frame.face_swapper.apply_post_processing", side_effect=lambda f, b: f)
    @patch("modules.processors.frame.face_swapper.batch_swap_faces")
    @patch("modules.processors.frame.face_swapper.swap_face")
    @patch("modules.processors.frame.face_swapper._build_pairs_live")
    def test_v2_skips_swap_when_no_pairs(self, mock_build, mock_swap, mock_batch, _mock_post):
        """process_frame_v2 does nothing when no valid pairs."""
        from modules.processors.frame.face_swapper import process_frame_v2
        import modules.globals
        modules.globals.opacity = 1.0
        modules.globals.target_path = None

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mock_build.return_value = []

        result = process_frame_v2(frame)

        mock_swap.assert_not_called()
        mock_batch.assert_not_called()

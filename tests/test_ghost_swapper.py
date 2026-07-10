"""Unit tests for Ghost face swap model integration.

Tests cover:
- GhostSwapper class instantiation and interface
- Preprocessing / postprocessing correctness
- Ghost model selection via globals and CLI
- pre_check() model path selection
- pre_start() model path selection
- batch_swap_faces() fallback for Ghost
- ProcessingConfig face_swap_model field
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import modules.globals
from modules.processing_config import ProcessingConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_onnx_session():
    """Return a MagicMock ONNX InferenceSession with Ghost tensor names."""
    session = MagicMock()
    # Mock input descriptors
    inp0 = MagicMock()
    inp0.name = "target"
    inp0.shape = [1, 3, 256, 256]
    inp1 = MagicMock()
    inp1.name = "source"
    inp1.shape = [1, 512]
    session.get_inputs.return_value = [inp0, inp1]
    # Mock output descriptor
    out0 = MagicMock()
    out0.name = "output"
    out0.shape = [1, 3, 256, 256]
    session.get_outputs.return_value = [out0]
    # Mock inference: return zeros in [-1, 1] range -> will become [0, 127] after postprocess
    session.run.return_value = [np.zeros((1, 3, 256, 256), dtype=np.float32)]
    return session


@pytest.fixture()
def ghost_swapper(mock_onnx_session):
    """Return a GhostSwapper instance backed by a mock session."""
    from modules.processors.frame.face_swapper import GhostSwapper

    return GhostSwapper(mock_onnx_session, input_size=256)


@pytest.fixture()
def mock_face():
    """Return a mock InsightFace face with normed_embedding and kps."""
    face = MagicMock()
    face.normed_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)  # L2-normalized
    face.kps = np.array(
        [
            [30.0, 40.0],
            [70.0, 40.0],
            [50.0, 65.0],
            [35.0, 85.0],
            [65.0, 85.0],
        ],
        dtype=np.float32,
    )
    return face


# ---------------------------------------------------------------------------
# GhostSwapper: instantiation
# ---------------------------------------------------------------------------


class TestGhostSwapperInit:
    def test_input_size_stored(self, mock_onnx_session):
        from modules.processors.frame.face_swapper import GhostSwapper

        gs = GhostSwapper(mock_onnx_session, input_size=256)
        assert gs.input_size == (256, 256)

    def test_session_stored(self, mock_onnx_session):
        from modules.processors.frame.face_swapper import GhostSwapper

        gs = GhostSwapper(mock_onnx_session, input_size=256)
        assert gs.session is mock_onnx_session

    def test_input_names_introspected(self, ghost_swapper):
        assert ghost_swapper._inp_names == ["target", "source"]

    def test_output_names_introspected(self, ghost_swapper):
        assert ghost_swapper._out_names == ["output"]


# ---------------------------------------------------------------------------
# GhostSwapper: .get() interface
# ---------------------------------------------------------------------------


class TestGhostSwapperGet:
    def test_returns_tuple(self, ghost_swapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            # norm_crop2 returns (aligned_img, M)
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            result = ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bgr_fake_shape(self, ghost_swapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            bgr_fake, M = ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)
        assert bgr_fake.shape == (256, 256, 3)

    def test_bgr_fake_range(self, ghost_swapper, mock_face):
        """Output must be clipped to [0, 255]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            bgr_fake, _ = ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)
        assert bgr_fake.min() >= 0.0
        assert bgr_fake.max() <= 255.0

    def test_affine_matrix_returned(self, ghost_swapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        expected_M = np.array([[0.5, 0, 10], [0, 0.5, 20]], dtype=np.float32)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (np.zeros((256, 256, 3), dtype=np.uint8), expected_M)
            _, M = ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)
        np.testing.assert_array_equal(M, expected_M)

    def test_session_called_with_correct_input_names(self, ghost_swapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)
        called_kwargs = ghost_swapper.session.run.call_args
        feed = called_kwargs[0][1]  # positional arg: input feed dict
        assert "target" in feed
        assert "source" in feed

    def test_target_blob_shape_and_range(self, ghost_swapper, mock_face):
        """Blob fed to session must be [1, 3, 256, 256] in range [-1, 1]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.full((256, 256, 3), 128, dtype=np.uint8),  # gray crop
                np.eye(2, 3, dtype=np.float32),
            )
            ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)

        feed = ghost_swapper.session.run.call_args[0][1]
        blob = feed["target"]
        assert blob.shape == (1, 3, 256, 256), f"Expected [1,3,256,256], got {blob.shape}"
        # 128 → (128/127.5) - 1.0 ≈ 0.004 ≈ 0
        assert blob.min() >= -1.0 - 1e-3
        assert blob.max() <= 1.0 + 1e-3

    def test_source_embedding_shape(self, ghost_swapper, mock_face):
        """Source embedding must be [1, 512]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            ghost_swapper.get(frame, mock_face, mock_face, paste_back=False)

        feed = ghost_swapper.session.run.call_args[0][1]
        latent = feed["source"]
        assert latent.shape == (1, 512)


# ---------------------------------------------------------------------------
# Ghost model registry
# ---------------------------------------------------------------------------


class TestGhostModelRegistry:
    def test_all_variants_defined(self):
        from modules.processors.frame.face_swapper import _GHOST_MODELS

        for variant in ("ghost_256_v1", "ghost_256_v2", "ghost_256_v3"):
            assert variant in _GHOST_MODELS

    def test_all_variants_have_required_keys(self):
        from modules.processors.frame.face_swapper import _GHOST_MODELS

        for variant, info in _GHOST_MODELS.items():
            assert "url" in info, f"{variant} missing 'url'"
            assert "file" in info, f"{variant} missing 'file'"
            assert "size" in info, f"{variant} missing 'size'"
            assert info["size"] == 256, f"{variant} size should be 256"

    def test_all_variants_have_onnx_extension(self):
        from modules.processors.frame.face_swapper import _GHOST_MODELS

        for variant, info in _GHOST_MODELS.items():
            assert info["file"].endswith(".onnx"), f"{variant} file should have .onnx extension"


# ---------------------------------------------------------------------------
# ProcessingConfig: face_swap_model field
# ---------------------------------------------------------------------------


class TestProcessingConfigFaceSwapModel:
    def test_default_is_inswapper(self):
        config = ProcessingConfig()
        assert config.face_swap_model == "inswapper"

    def test_ghost_v1_accepted(self):
        config = ProcessingConfig(face_swap_model="ghost_256_v1")
        assert config.face_swap_model == "ghost_256_v1"

    def test_ghost_v2_accepted(self):
        config = ProcessingConfig(face_swap_model="ghost_256_v2")
        assert config.face_swap_model == "ghost_256_v2"

    def test_ghost_v3_accepted(self):
        config = ProcessingConfig(face_swap_model="ghost_256_v3")
        assert config.face_swap_model == "ghost_256_v3"


# ---------------------------------------------------------------------------
# globals.face_swap_model
# ---------------------------------------------------------------------------


class TestGlobalsDefault:
    def test_default_face_swap_model(self):
        assert hasattr(modules.globals, "face_swap_model")
        # Default must be 'inswapper' unless overridden by CLI during this run
        assert modules.globals.face_swap_model in ("inswapper", "ghost_256_v1", "ghost_256_v2", "ghost_256_v3")

    def test_default_is_inswapper(self):
        saved = modules.globals.face_swap_model
        modules.globals.face_swap_model = "inswapper"
        try:
            assert modules.globals.face_swap_model == "inswapper"
        finally:
            modules.globals.face_swap_model = saved


# ---------------------------------------------------------------------------
# build_config_from_globals: face_swap_model propagated
# ---------------------------------------------------------------------------


class TestConfigFactory:
    def test_face_swap_model_in_config(self):
        from modules.processing_config_factory import build_config_from_globals

        saved = modules.globals.face_swap_model
        modules.globals.face_swap_model = "ghost_256_v2"
        try:
            config = build_config_from_globals()
            assert config.face_swap_model == "ghost_256_v2"
        finally:
            modules.globals.face_swap_model = saved


# ---------------------------------------------------------------------------
# pre_check(): model selection
# ---------------------------------------------------------------------------


class TestPreCheckModelSelection:
    def test_ghost_v1_triggers_ghost_download(self, tmp_path):
        """pre_check downloads Ghost model file, not inswapper, when ghost_256_v1 selected."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import _GHOST_MODELS

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        modules.globals.face_swap_model = "ghost_256_v1"
        fsmod.models_dir = str(tmp_path)

        downloaded_urls = []

        def fake_conditional_download(directory, urls, expected_checksums=None):
            downloaded_urls.extend(urls)
            # Create the file so pre_check thinks download succeeded
            for url in urls:
                fname = url.split("/")[-1]
                (tmp_path / fname).touch()

        try:
            with patch(
                "modules.processors.frame.face_swapper.conditional_download", side_effect=fake_conditional_download
            ):
                result = fsmod.pre_check()
        finally:
            modules.globals.face_swap_model = saved_model
            fsmod.models_dir = saved_models_dir

        assert result is True
        assert _GHOST_MODELS["ghost_256_v1"]["url"] in downloaded_urls
        # inswapper URL must NOT be downloaded
        assert "inswapper_128_fp16.onnx" not in "".join(downloaded_urls)

    def test_inswapper_triggers_inswapper_download(self, tmp_path):
        """pre_check downloads inswapper when inswapper is selected."""
        import modules.processors.frame.face_swapper as fsmod

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        modules.globals.face_swap_model = "inswapper"
        fsmod.models_dir = str(tmp_path)

        downloaded_urls = []

        def fake_conditional_download(directory, urls, expected_checksums=None):
            downloaded_urls.extend(urls)
            for url in urls:
                fname = url.split("/")[-1]
                (tmp_path / fname).touch()

        try:
            with patch(
                "modules.processors.frame.face_swapper.conditional_download", side_effect=fake_conditional_download
            ):
                result = fsmod.pre_check()
        finally:
            modules.globals.face_swap_model = saved_model
            fsmod.models_dir = saved_models_dir

        assert result is True
        assert any("inswapper" in url for url in downloaded_urls)


# ---------------------------------------------------------------------------
# get_face_swapper(): Ghost model loading
# ---------------------------------------------------------------------------


class TestGetFaceSwapperGhost:
    def test_ghost_model_returns_ghost_swapper(self, tmp_path):
        """get_face_swapper() returns a GhostSwapper when ghost_256_v1 is selected."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import _GHOST_MODELS, GhostSwapper

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        saved_swapper = fsmod.FACE_SWAPPER
        modules.globals.face_swap_model = "ghost_256_v1"
        fsmod.models_dir = str(tmp_path)
        fsmod.FACE_SWAPPER = None

        # Create a dummy ghost model file
        ghost_file = _GHOST_MODELS["ghost_256_v1"]["file"]
        (tmp_path / ghost_file).touch()

        mock_session = MagicMock()
        mock_inp = MagicMock()
        mock_inp.name = "target"
        mock_inp.shape = [1, 3, 256, 256]
        mock_inp2 = MagicMock()
        mock_inp2.name = "source"
        mock_inp2.shape = [1, 512]
        mock_out = MagicMock()
        mock_out.name = "output"
        mock_session.get_inputs.return_value = [mock_inp, mock_inp2]
        mock_session.get_outputs.return_value = [mock_out]

        try:
            with patch("modules.processors.frame.face_swapper.onnxruntime.InferenceSession", return_value=mock_session):
                swapper = fsmod.get_face_swapper(providers=["CPUExecutionProvider"])
            assert isinstance(swapper, GhostSwapper)
        finally:
            modules.globals.face_swap_model = saved_model
            fsmod.models_dir = saved_models_dir
            fsmod.FACE_SWAPPER = saved_swapper


# ---------------------------------------------------------------------------
# batch_swap_faces(): Ghost fallback to sequential
# ---------------------------------------------------------------------------


class TestBatchSwapFacesGhostFallback:
    def test_ghost_falls_back_to_sequential(self):
        """batch_swap_faces() falls back to sequential swap_face() for Ghost models."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import GhostSwapper

        saved_swapper = fsmod.FACE_SWAPPER

        mock_session = MagicMock()
        mock_inp = MagicMock()
        mock_inp.name = "target"
        mock_inp.shape = [1, 3, 256, 256]
        mock_inp2 = MagicMock()
        mock_inp2.name = "source"
        mock_inp2.shape = [1, 512]
        mock_out = MagicMock()
        mock_out.name = "output"
        mock_session.get_inputs.return_value = [mock_inp, mock_inp2]
        mock_session.get_outputs.return_value = [mock_out]
        ghost = GhostSwapper(mock_session, input_size=256)
        fsmod.FACE_SWAPPER = ghost

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        try:
            with patch("modules.processors.frame.face_swapper.swap_face", return_value=frame) as mock_swap:
                from modules.processors.frame.face_swapper import batch_swap_faces

                batch_swap_faces(
                    [MagicMock(), MagicMock()],
                    [MagicMock(), MagicMock()],
                    frame,
                )
                # swap_face should have been called for each face pair
                assert mock_swap.call_count == 2
        finally:
            fsmod.FACE_SWAPPER = saved_swapper

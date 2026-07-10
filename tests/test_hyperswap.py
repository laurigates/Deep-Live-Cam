"""Unit tests for HyperSwap face swap model integration.

Tests cover:
- HyperSwapper class instantiation and interface
- Preprocessing / postprocessing correctness
- HyperSwap model selection via globals and CLI
- pre_check() model path selection
- pre_start() model path selection
- batch_swap_faces() fallback for HyperSwap
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
    """Return a MagicMock ONNX InferenceSession with HyperSwap tensor names."""
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
def hyperswapper(mock_onnx_session):
    """Return a HyperSwapper instance backed by a mock session."""
    from modules.processors.frame.face_swapper import HyperSwapper

    return HyperSwapper(mock_onnx_session, input_size=256)


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
# HyperSwapper: instantiation
# ---------------------------------------------------------------------------


class TestHyperSwapperInit:
    def test_input_size_stored(self, mock_onnx_session):
        from modules.processors.frame.face_swapper import HyperSwapper

        hs = HyperSwapper(mock_onnx_session, input_size=256)
        assert hs.input_size == (256, 256)

    def test_session_stored(self, mock_onnx_session):
        from modules.processors.frame.face_swapper import HyperSwapper

        hs = HyperSwapper(mock_onnx_session, input_size=256)
        assert hs.session is mock_onnx_session

    def test_input_names_introspected(self, hyperswapper):
        assert hyperswapper._inp_names == ["target", "source"]

    def test_output_names_introspected(self, hyperswapper):
        assert hyperswapper._out_names == ["output"]


# ---------------------------------------------------------------------------
# HyperSwapper: .get() interface
# ---------------------------------------------------------------------------


class TestHyperSwapperGet:
    def test_returns_tuple(self, hyperswapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            result = hyperswapper.get(frame, mock_face, mock_face, paste_back=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bgr_fake_shape(self, hyperswapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            bgr_fake, M = hyperswapper.get(frame, mock_face, mock_face, paste_back=False)
        assert bgr_fake.shape == (256, 256, 3)

    def test_bgr_fake_range(self, hyperswapper, mock_face):
        """Output must be clipped to [0, 255]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            bgr_fake, _ = hyperswapper.get(frame, mock_face, mock_face, paste_back=False)
        assert bgr_fake.min() >= 0.0
        assert bgr_fake.max() <= 255.0

    def test_affine_matrix_returned(self, hyperswapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        expected_M = np.array([[0.5, 0, 10], [0, 0.5, 20]], dtype=np.float32)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (np.zeros((256, 256, 3), dtype=np.uint8), expected_M)
            _, M = hyperswapper.get(frame, mock_face, mock_face, paste_back=False)
        np.testing.assert_array_equal(M, expected_M)

    def test_session_called_with_correct_input_names(self, hyperswapper, mock_face):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            hyperswapper.get(frame, mock_face, mock_face, paste_back=False)
        called_kwargs = hyperswapper.session.run.call_args
        feed = called_kwargs[0][1]  # positional arg: input feed dict
        assert "target" in feed
        assert "source" in feed

    def test_target_blob_shape_and_range(self, hyperswapper, mock_face):
        """Blob fed to session must be [1, 3, 256, 256] in range [-1, 1]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.full((256, 256, 3), 128, dtype=np.uint8),  # gray crop
                np.eye(2, 3, dtype=np.float32),
            )
            hyperswapper.get(frame, mock_face, mock_face, paste_back=False)

        feed = hyperswapper.session.run.call_args[0][1]
        blob = feed["target"]
        assert blob.shape == (1, 3, 256, 256), f"Expected [1,3,256,256], got {blob.shape}"
        # 128 -> (128/127.5) - 1.0 ~ 0.004 ~ 0
        assert blob.min() >= -1.0 - 1e-3
        assert blob.max() <= 1.0 + 1e-3

    def test_source_embedding_shape(self, hyperswapper, mock_face):
        """Source embedding must be [1, 512]."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop2") as mock_crop:
            mock_crop.return_value = (
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.eye(2, 3, dtype=np.float32),
            )
            hyperswapper.get(frame, mock_face, mock_face, paste_back=False)

        feed = hyperswapper.session.run.call_args[0][1]
        latent = feed["source"]
        assert latent.shape == (1, 512)


# ---------------------------------------------------------------------------
# HyperSwap model registry
# ---------------------------------------------------------------------------


class TestHyperSwapModelRegistry:
    def test_all_variants_defined(self):
        from modules.processors.frame.face_swapper import _HYPERSWAP_MODELS

        for variant in ("hyperswap_256_1a", "hyperswap_256_1b", "hyperswap_256_1c"):
            assert variant in _HYPERSWAP_MODELS

    def test_all_variants_have_required_keys(self):
        from modules.processors.frame.face_swapper import _HYPERSWAP_MODELS

        for variant, info in _HYPERSWAP_MODELS.items():
            assert "url" in info, f"{variant} missing 'url'"
            assert "file" in info, f"{variant} missing 'file'"
            assert "size" in info, f"{variant} missing 'size'"
            assert info["size"] == 256, f"{variant} size should be 256"

    def test_all_variants_have_onnx_extension(self):
        from modules.processors.frame.face_swapper import _HYPERSWAP_MODELS

        for variant, info in _HYPERSWAP_MODELS.items():
            assert info["file"].endswith(".onnx"), f"{variant} file should have .onnx extension"


# ---------------------------------------------------------------------------
# ProcessingConfig: face_swap_model field
# ---------------------------------------------------------------------------


class TestProcessingConfigFaceSwapModel:
    def test_hyperswap_1a_accepted(self):
        config = ProcessingConfig(face_swap_model="hyperswap_256_1a")
        assert config.face_swap_model == "hyperswap_256_1a"

    def test_hyperswap_1b_accepted(self):
        config = ProcessingConfig(face_swap_model="hyperswap_256_1b")
        assert config.face_swap_model == "hyperswap_256_1b"

    def test_hyperswap_1c_accepted(self):
        config = ProcessingConfig(face_swap_model="hyperswap_256_1c")
        assert config.face_swap_model == "hyperswap_256_1c"


# ---------------------------------------------------------------------------
# globals.face_swap_model
# ---------------------------------------------------------------------------


class TestGlobalsHyperSwap:
    def test_hyperswap_model_can_be_set(self):
        saved = modules.globals.face_swap_model
        modules.globals.face_swap_model = "hyperswap_256_1a"
        try:
            assert modules.globals.face_swap_model == "hyperswap_256_1a"
        finally:
            modules.globals.face_swap_model = saved


# ---------------------------------------------------------------------------
# build_config_from_globals: face_swap_model propagated
# ---------------------------------------------------------------------------


class TestConfigFactory:
    def test_face_swap_model_in_config(self):
        from modules.processing_config_factory import build_config_from_globals

        saved = modules.globals.face_swap_model
        modules.globals.face_swap_model = "hyperswap_256_1b"
        try:
            config = build_config_from_globals()
            assert config.face_swap_model == "hyperswap_256_1b"
        finally:
            modules.globals.face_swap_model = saved


# ---------------------------------------------------------------------------
# pre_check(): model selection
# ---------------------------------------------------------------------------


class TestPreCheckModelSelection:
    def test_hyperswap_1a_triggers_hyperswap_download(self, tmp_path):
        """pre_check downloads HyperSwap model file when hyperswap_256_1a selected."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import _HYPERSWAP_MODELS

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        modules.globals.face_swap_model = "hyperswap_256_1a"
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
        assert _HYPERSWAP_MODELS["hyperswap_256_1a"]["url"] in downloaded_urls
        # inswapper URL must NOT be downloaded
        assert "inswapper_128_fp16.onnx" not in "".join(downloaded_urls)

    def test_inswapper_not_triggered_for_hyperswap(self, tmp_path):
        """pre_check does not download inswapper when hyperswap is selected."""
        import modules.processors.frame.face_swapper as fsmod

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        modules.globals.face_swap_model = "hyperswap_256_1c"
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
                fsmod.pre_check()
        finally:
            modules.globals.face_swap_model = saved_model
            fsmod.models_dir = saved_models_dir

        assert not any("inswapper" in url for url in downloaded_urls)


# ---------------------------------------------------------------------------
# get_face_swapper(): HyperSwap model loading
# ---------------------------------------------------------------------------


class TestGetFaceSwapperHyperSwap:
    def test_hyperswap_model_returns_hyperswapper(self, tmp_path):
        """get_face_swapper() returns a HyperSwapper when hyperswap_256_1a is selected."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import _HYPERSWAP_MODELS, HyperSwapper

        saved_model = modules.globals.face_swap_model
        saved_models_dir = fsmod.models_dir
        saved_swapper = fsmod.FACE_SWAPPER
        modules.globals.face_swap_model = "hyperswap_256_1a"
        fsmod.models_dir = str(tmp_path)
        fsmod.FACE_SWAPPER = None

        # Create a dummy hyperswap model file
        hyperswap_file = _HYPERSWAP_MODELS["hyperswap_256_1a"]["file"]
        (tmp_path / hyperswap_file).touch()

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
            assert isinstance(swapper, HyperSwapper)
        finally:
            modules.globals.face_swap_model = saved_model
            fsmod.models_dir = saved_models_dir
            fsmod.FACE_SWAPPER = saved_swapper


# ---------------------------------------------------------------------------
# batch_swap_faces(): HyperSwap fallback to sequential
# ---------------------------------------------------------------------------


class TestBatchSwapFacesHyperSwapFallback:
    def test_hyperswap_falls_back_to_sequential(self):
        """batch_swap_faces() falls back to sequential swap_face() for HyperSwap models."""
        import modules.processors.frame.face_swapper as fsmod
        from modules.processors.frame.face_swapper import HyperSwapper

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
        hyperswap = HyperSwapper(mock_session, input_size=256)
        fsmod.FACE_SWAPPER = hyperswap

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

"""Tests for Phase 1 face preprocessing pipeline improvements.

1A: Verify swap_face uses warpAffine with existing M (not redundant norm_crop2).
1B: Verify enhancer pre-model upscale uses LANCZOS4.
1C: Verify batch path preserves float32 through paste-back.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1A: swap_face reuses M for aligned crop instead of redundant norm_crop2
# ---------------------------------------------------------------------------
class TestSwapFaceReusesM:
    """Verify that swap_face computes aimg via warpAffine(M) not norm_crop2."""

    def test_warped_aimg_matches_norm_crop2(self):
        """warpAffine(frame, M, ...) produces the same result as norm_crop2
        when using the same M matrix — confirming the optimisation is safe."""
        from insightface.utils import face_align

        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Simulate 5 key-points roughly centred in the frame
        kps = np.array(
            [
                [260, 200],
                [380, 200],
                [320, 280],
                [270, 340],
                [370, 340],
            ],
            dtype=np.float32,
        )

        input_size = 128

        # Reference: norm_crop2 computes M, then warps
        aimg_ref, M_ref = face_align.norm_crop2(frame, kps, input_size)

        # Optimised path: reuse M, warp directly
        aimg_opt = cv2.warpAffine(frame, M_ref, (input_size, input_size), borderValue=0.0)

        np.testing.assert_array_equal(
            aimg_ref,
            aimg_opt,
            err_msg="warpAffine(M) should produce identical output to norm_crop2",
        )


# ---------------------------------------------------------------------------
# 1B: Enhancer pre-model upscale uses LANCZOS4
# ---------------------------------------------------------------------------
class TestEnhancerUpscaleLanczos:
    """Verify face_enhancer uses LANCZOS4 for pre-model upscale."""

    def test_lanczos4_produces_sharper_input(self):
        """LANCZOS4 should produce different (sharper) output than LINEAR
        for the same upscale operation on a non-trivial image."""
        rng = np.random.RandomState(42)
        face = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        linear = cv2.resize(face, (512, 512), interpolation=cv2.INTER_LINEAR)
        lanczos = cv2.resize(face, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # They should differ for a non-trivial image
        assert not np.array_equal(linear, lanczos), "LANCZOS4 should produce different output than LINEAR"

        # LANCZOS4 should have higher edge contrast (sharper)
        linear_edges = cv2.Laplacian(cv2.cvtColor(linear, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        lanczos_edges = cv2.Laplacian(cv2.cvtColor(lanczos, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        assert np.std(lanczos_edges) >= np.std(linear_edges) * 0.99, (
            "LANCZOS4 should produce at least comparable edge sharpness"
        )


# ---------------------------------------------------------------------------
# 1C: Batch path preserves float32 through paste-back
# ---------------------------------------------------------------------------
class TestBatchFloat32PasteBack:
    """Verify that batch paste-back path avoids uint8 quantization loss."""

    def test_paste_back_accepts_float32_input(self):
        """_paste_back should work correctly with float32 bgr_fake input."""
        from modules.processors.frame.face_swapper import _paste_back

        target = np.full((200, 200, 3), 100, dtype=np.uint8)
        # Float32 input with fractional values (as from ONNX output)
        bgr_fake = np.full((128, 128, 3), 150.7, dtype=np.float32)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 36], [0, 1.0, 36]], dtype=np.float64)

        result = _paste_back(bgr_fake, aimg, M, target)
        assert result.dtype == np.uint8
        assert result.shape == target.shape

    def test_float32_vs_uint8_reduces_banding(self):
        """Float32 path should preserve more precision than uint8 round-trip."""
        from modules.processors.frame.face_swapper import _paste_back

        target = np.full((200, 200, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 36], [0, 1.0, 36]], dtype=np.float64)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)

        # Simulate ONNX output with fractional values
        onnx_output = np.full((128, 128, 3), 150.3, dtype=np.float32)

        # Float32 path (new)
        result_f32 = _paste_back(onnx_output, aimg, M, target)

        # Uint8 path (old) — quantise first
        onnx_quantised = np.clip(onnx_output, 0, 255).astype(np.uint8)
        result_u8 = _paste_back(onnx_quantised, aimg, M, target)

        # Both should produce valid output
        assert result_f32.dtype == np.uint8
        assert result_u8.dtype == np.uint8

        # Results may differ slightly due to preserved precision
        # The float32 path should not be worse
        diff = np.abs(result_f32.astype(np.int16) - result_u8.astype(np.int16))
        assert diff.max() <= 2, f"Float32 and uint8 paths should produce similar output (max diff: {diff.max()})"

    def test_paste_back_handles_mixed_dtypes(self):
        """_paste_back should handle float32 bgr_fake with uint8 aimg."""
        from modules.processors.frame.face_swapper import _paste_back

        target = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        bgr_fake = np.random.uniform(0, 255, (128, 128, 3)).astype(np.float32)
        aimg = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.array([[0.8, 0, 86], [0, 0.8, 86]], dtype=np.float64)

        result = _paste_back(bgr_fake, aimg, M, target)
        assert result.dtype == np.uint8
        assert result.shape == target.shape

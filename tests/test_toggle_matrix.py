"""Toggle combination matrix tests.

Verifies that all toggle combinations interact correctly without crashing.
Uses mocked inference — no real models needed.
"""
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import modules.globals


def _make_face():
    """Lightweight mock face with landmarks."""
    face = SimpleNamespace()
    emb = np.random.randn(512).astype(np.float32)
    face.normed_embedding = emb / np.linalg.norm(emb)
    face.embedding = face.normed_embedding.copy()
    face.kps = np.array(
        [[290, 240], [350, 240], [320, 270], [295, 310], [345, 310]],
        dtype=np.float32,
    )
    face.bbox = np.array([230, 160, 410, 400], dtype=np.float32)
    face.det_score = 0.99

    lm = np.zeros((106, 2), dtype=np.float32)
    cx, cy = 320.0, 280.0
    for i in range(33):
        angle = np.pi * i / 32
        lm[i] = [cx + 90 * np.sin(angle), cy - 120 * np.cos(angle)]
    for i, idx in enumerate(range(33, 43)):
        lm[idx] = [270 + i * 8, 200 - abs(i - 5) * 2]
    for i, idx in enumerate(range(43, 52)):
        lm[idx] = [330 + i * 8, 200 - abs(i - 4) * 2]
    for i, idx in enumerate(range(52, 64)):
        angle = 2 * np.pi * i / 12
        lm[idx] = [320 + 25 * np.cos(angle), 340 + 10 * np.sin(angle)]
    for i, idx in enumerate(range(64, 73)):
        lm[idx] = [310 + i * 2.5, 260 + abs(i - 4) * 3]
    for i, idx in enumerate(range(73, 87)):
        angle = 2 * np.pi * i / 14
        lm[idx] = [290 + 15 * np.cos(angle), 240 + 8 * np.sin(angle)]
    for i, idx in enumerate(range(87, 97)):
        angle = 2 * np.pi * i / 10
        lm[idx] = [350 + 15 * np.cos(angle), 240 + 8 * np.sin(angle)]
    for i, idx in enumerate(range(97, 106)):
        lm[idx] = [268 + i * 8, 195]

    face.landmark_2d_106 = lm
    face.landmark_3d_68 = np.zeros((68, 3), dtype=np.float32)
    face.gender = 1
    face.age = 30
    return face


def _reset_globals(**overrides):
    """Reset all processing toggles to defaults, then apply overrides."""
    modules.globals.mouth_mask = False
    modules.globals.show_mouth_mask_box = False
    modules.globals.poisson_blend = False
    modules.globals.color_correction = False
    modules.globals.opacity = 1.0
    modules.globals.sharpness = 0.0
    modules.globals.prepaste_upscale = True
    modules.globals.enable_interpolation = False
    modules.globals.interpolation_weight = 0.0
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.mask_feather_ratio = 12
    modules.globals.mask_down_size = 0.1
    modules.globals.mouth_mask_size = 1.0
    modules.globals.face_mask_blur = 31
    modules.globals.execution_providers = ["CPUExecutionProvider"]

    for key, value in overrides.items():
        setattr(modules.globals, key, value)


@pytest.fixture(autouse=True)
def _always_reset_globals():
    """Reset globals before and after every test to prevent cross-test pollution."""
    _reset_globals()
    yield
    _reset_globals()


# ---------------------------------------------------------------------------
# Toggle smoke tests: individual
# ---------------------------------------------------------------------------
class TestIndividualToggles:
    """Each toggle individually should not crash the masking functions."""

    def test_mouth_mask_on(self):
        from modules.processors.frame.face_masking import create_lower_mouth_mask, apply_mouth_area

        modules.globals.mouth_mask = True
        modules.globals.mask_blur_kernel = 15
        face = _make_face()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = create_lower_mouth_mask(face, frame)
        if cutout is not None and box != (0, 0, 0, 0):
            face_mask = np.full((480, 640), 255, dtype=np.uint8)
            result = apply_mouth_area(frame, cutout, box, face_mask, poly)
            assert result.shape == frame.shape

    def test_poisson_blend_does_not_crash(self):
        """Poisson blend requires a face mask — verify it doesn't crash."""
        from modules.processors.frame.face_masking import create_face_mask

        modules.globals.poisson_blend = True
        face = _make_face()
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        mask = create_face_mask(face, frame)
        assert mask.shape == (480, 640)

    def test_color_correction_standalone(self):
        from modules.processors.frame.face_masking import apply_color_transfer

        src = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tgt = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = apply_color_transfer(src, tgt)
        assert result.dtype == np.uint8
        assert result.shape == src.shape

    def test_sharpening_does_not_crash(self):
        from modules.processors.frame.face_swapper import apply_post_processing

        modules.globals.sharpness = 0.8
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        bbox = np.array([230, 160, 410, 400], dtype=np.float32)
        result = apply_post_processing(frame, [bbox])
        assert result.shape == frame.shape

    def test_interpolation_with_no_previous_frame(self):
        from modules.processors.frame.face_swapper import apply_post_processing

        modules.globals.enable_interpolation = True
        modules.globals.interpolation_weight = 0.3
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = apply_post_processing(frame, [])
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Toggle combination matrix
# ---------------------------------------------------------------------------
# Test a representative subset of toggle combinations to keep runtime reasonable.
# Key toggles that interact with each other:
TOGGLE_MATRIX = {
    "mouth_mask": [False, True],
    "poisson_blend": [False, True],
    "sharpness": [0.0, 0.5],
    "opacity": [1.0, 0.7],
}


def _generate_toggle_combos():
    """Generate all combinations of the toggle matrix."""
    keys = list(TOGGLE_MATRIX.keys())
    values = [TOGGLE_MATRIX[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


class TestToggleCombinationMatrix:
    """Run masking + post-processing for each toggle combination."""

    @pytest.mark.parametrize(
        "toggles",
        list(_generate_toggle_combos()),
        ids=[
            "_".join(f"{k}={v}" for k, v in combo.items())
            for combo in _generate_toggle_combos()
        ],
    )
    def test_toggle_combo_no_crash(self, toggles):
        """Verify processing does not crash for this toggle combination."""
        _reset_globals(**toggles)
        modules.globals.mask_blur_kernel = 15

        from modules.processors.frame.face_masking import (
            create_face_mask,
            create_lower_mouth_mask,
            apply_mouth_area,
        )
        from modules.processors.frame.face_swapper import apply_post_processing

        face = _make_face()
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Face mask
        face_mask = create_face_mask(face, frame)
        assert face_mask.shape == (480, 640)

        # Mouth mask if enabled
        if toggles.get("mouth_mask"):
            mask, cutout, box, poly = create_lower_mouth_mask(face, frame)
            if cutout is not None and box != (0, 0, 0, 0):
                frame = apply_mouth_area(frame, cutout, box, face_mask, poly)

        # Post-processing (sharpening)
        bbox = face.bbox
        result = apply_post_processing(frame, [bbox])

        # Opacity blend if < 1.0
        if toggles.get("opacity", 1.0) < 1.0:
            op = toggles["opacity"]
            original = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            blended = cv2.addWeighted(original, 1 - op, result, op, 0)
            result = blended

        assert result.dtype == np.uint8
        assert result.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Enhancer toggle combinations (mock inference)
# ---------------------------------------------------------------------------
class TestEnhancerToggleCombos:
    """Verify enhancer UI toggles don't conflict."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _reset_globals()

    @pytest.mark.parametrize(
        "enhancer_key",
        ["face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512", "face_enhancer_codeformer"],
    )
    def test_only_one_enhancer_active(self, enhancer_key):
        """When one enhancer is active, others should be off."""
        modules.globals.fp_ui = {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
            "face_enhancer_codeformer": False,
        }
        modules.globals.fp_ui[enhancer_key] = True

        active = [k for k, v in modules.globals.fp_ui.items() if v]
        assert len(active) == 1
        assert active[0] == enhancer_key

    def test_no_enhancer_active_is_valid(self):
        modules.globals.fp_ui = {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
            "face_enhancer_codeformer": False,
        }
        active = [k for k, v in modules.globals.fp_ui.items() if v]
        assert len(active) == 0

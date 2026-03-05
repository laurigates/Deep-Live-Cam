"""Tests for SHA-256 checksum verification on model downloads — Issue #88.

Verifies that all conditional_download() calls for known model files pass
a non-empty expected_checksums dict so downloaded models are integrity-checked.

Strategy: patch conditional_download at the module level and invoke pre_check()
for each processor, then inspect the captured call arguments.
"""

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_conditional_download_calls(module_path: str, pre_check_fn, setup_fn=None):
    """Run pre_check_fn with conditional_download patched; return captured calls.

    *setup_fn* is an optional callable that receives the mock and can configure
    side-effects before pre_check_fn is called (e.g. os.path.exists stubs).
    """
    captured: list[dict] = []

    def fake_download(directory, urls, expected_checksums=None):
        captured.append(
            {
                "directory": directory,
                "urls": urls,
                "expected_checksums": expected_checksums,
            }
        )

    with (
        patch(f"{module_path}.conditional_download", side_effect=fake_download),
        patch(f"{module_path}.os.path.exists", return_value=True),
        patch(f"{module_path}.os.makedirs", return_value=None),
    ):
        if setup_fn:
            setup_fn()
        pre_check_fn()

    return captured


# ---------------------------------------------------------------------------
# face_swapper — inswapper model
# ---------------------------------------------------------------------------


class TestFaceSwapperChecksums:
    def test_inswapper_download_includes_expected_checksums(self):
        """pre_check() for inswapper must pass expected_checksums to conditional_download."""
        import modules.globals
        import modules.processors.frame.face_swapper as face_swapper

        original_model = getattr(modules.globals, "face_swap_model", "inswapper")
        modules.globals.face_swap_model = "inswapper"

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append(
                {
                    "directory": directory,
                    "urls": urls,
                    "expected_checksums": expected_checksums,
                }
            )

        try:
            with (
                patch("modules.processors.frame.face_swapper.conditional_download", side_effect=fake_download),
                patch("modules.processors.frame.face_swapper.os.path.exists", return_value=True),
                patch("modules.processors.frame.face_swapper.os.makedirs", return_value=None),
            ):
                face_swapper.pre_check()
        finally:
            modules.globals.face_swap_model = original_model

        assert len(captured) >= 1, "conditional_download was not called for inswapper"
        inswapper_call = captured[0]
        assert inswapper_call["expected_checksums"] is not None, (
            "inswapper download did not pass expected_checksums — downloaded model is not integrity-checked"
        )
        assert len(inswapper_call["expected_checksums"]) > 0, "inswapper expected_checksums dict is empty"
        assert "inswapper_128_fp16.onnx" in inswapper_call["expected_checksums"], (
            "expected_checksums does not contain entry for 'inswapper_128_fp16.onnx'"
        )

    def test_ghost_download_includes_expected_checksums(self):
        """pre_check() for ghost_256_v1 must pass expected_checksums to conditional_download."""
        import modules.globals
        import modules.processors.frame.face_swapper as face_swapper

        original_model = getattr(modules.globals, "face_swap_model", "inswapper")
        modules.globals.face_swap_model = "ghost_256_v1"

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append(
                {
                    "directory": directory,
                    "urls": urls,
                    "expected_checksums": expected_checksums,
                }
            )

        try:
            with (
                patch("modules.processors.frame.face_swapper.conditional_download", side_effect=fake_download),
                patch("modules.processors.frame.face_swapper.os.path.exists", return_value=True),
                patch("modules.processors.frame.face_swapper.os.makedirs", return_value=None),
            ):
                face_swapper.pre_check()
        finally:
            modules.globals.face_swap_model = original_model

        assert len(captured) >= 1, "conditional_download was not called for ghost_256_v1"
        ghost_call = captured[0]
        assert ghost_call["expected_checksums"] is not None, "ghost_256_v1 download did not pass expected_checksums"
        assert "ghost_256_v1.onnx" in ghost_call["expected_checksums"], (
            "expected_checksums does not contain entry for 'ghost_256_v1.onnx'"
        )

    def test_hyperswap_download_includes_expected_checksums(self):
        """pre_check() for hyperswap_256_1a must pass expected_checksums."""
        import modules.globals
        import modules.processors.frame.face_swapper as face_swapper

        original_model = getattr(modules.globals, "face_swap_model", "inswapper")
        modules.globals.face_swap_model = "hyperswap_256_1a"

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append(
                {
                    "directory": directory,
                    "urls": urls,
                    "expected_checksums": expected_checksums,
                }
            )

        try:
            with (
                patch("modules.processors.frame.face_swapper.conditional_download", side_effect=fake_download),
                patch("modules.processors.frame.face_swapper.os.path.exists", return_value=True),
                patch("modules.processors.frame.face_swapper.os.makedirs", return_value=None),
            ):
                face_swapper.pre_check()
        finally:
            modules.globals.face_swap_model = original_model

        assert len(captured) >= 1, "conditional_download was not called for hyperswap_256_1a"
        hs_call = captured[0]
        assert hs_call["expected_checksums"] is not None, "hyperswap_256_1a download did not pass expected_checksums"
        assert "hyperswap_1a_256.onnx" in hs_call["expected_checksums"], (
            "expected_checksums does not contain entry for 'hyperswap_1a_256.onnx'"
        )


# ---------------------------------------------------------------------------
# face_enhancer — GFPGAN model
# ---------------------------------------------------------------------------


class TestFaceEnhancerChecksums:
    def test_gfpgan_download_includes_expected_checksums(self):
        """pre_check() for face_enhancer must pass expected_checksums to conditional_download."""
        import modules.processors.frame.face_enhancer as face_enhancer

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append(
                {
                    "directory": directory,
                    "urls": urls,
                    "expected_checksums": expected_checksums,
                }
            )

        with (
            patch("modules.processors.frame.face_enhancer.conditional_download", side_effect=fake_download),
            patch("modules.processors.frame.face_enhancer.os.path.exists", return_value=False),
        ):
            face_enhancer.pre_check()

        assert len(captured) >= 1, "conditional_download was not called for GFPGAN"
        gfpgan_call = captured[0]
        assert gfpgan_call["expected_checksums"] is not None, (
            "GFPGAN download did not pass expected_checksums — downloaded model is not integrity-checked"
        )
        assert len(gfpgan_call["expected_checksums"]) > 0, "GFPGAN expected_checksums dict is empty"
        assert "gfpgan-1024.onnx" in gfpgan_call["expected_checksums"], (
            "expected_checksums does not contain entry for 'gfpgan-1024.onnx'"
        )


# ---------------------------------------------------------------------------
# face_occluder — XSeg model
# ---------------------------------------------------------------------------


class TestFaceOccluderChecksums:
    def test_xseg_download_includes_expected_checksums(self):
        """pre_check() for face_occluder must pass expected_checksums to conditional_download."""
        from modules import face_occluder

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append(
                {
                    "directory": directory,
                    "urls": urls,
                    "expected_checksums": expected_checksums,
                }
            )

        with (
            patch("modules.face_occluder.conditional_download", side_effect=fake_download),
            patch("modules.face_occluder.os.path.exists", return_value=False),
        ):
            face_occluder.pre_check()

        assert len(captured) >= 1, "conditional_download was not called for XSeg"
        xseg_call = captured[0]
        assert xseg_call["expected_checksums"] is not None, (
            "XSeg download did not pass expected_checksums — downloaded model is not integrity-checked"
        )
        assert len(xseg_call["expected_checksums"]) > 0, "XSeg expected_checksums dict is empty"
        assert "xseg_2.onnx" in xseg_call["expected_checksums"], (
            "expected_checksums does not contain entry for 'xseg_2.onnx'"
        )


# ---------------------------------------------------------------------------
# Checksum format validation
# ---------------------------------------------------------------------------


class TestChecksumFormat:
    """Validate that checksum values look like valid SHA-256 hex digests."""

    def _is_valid_sha256(self, value: str) -> bool:
        """Return True if value is a 64-character lowercase hex string."""
        if not isinstance(value, str):
            return False
        if len(value) != 64:
            return False
        try:
            int(value, 16)
            return True
        except ValueError:
            return False

    def test_inswapper_checksum_is_valid_sha256_format(self):
        """Checksum for inswapper_128_fp16.onnx must be a valid SHA-256 hex string."""
        import modules.globals
        import modules.processors.frame.face_swapper as face_swapper

        original_model = getattr(modules.globals, "face_swap_model", "inswapper")
        modules.globals.face_swap_model = "inswapper"

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append({"expected_checksums": expected_checksums})

        try:
            with (
                patch("modules.processors.frame.face_swapper.conditional_download", side_effect=fake_download),
                patch("modules.processors.frame.face_swapper.os.path.exists", return_value=True),
                patch("modules.processors.frame.face_swapper.os.makedirs", return_value=None),
            ):
                face_swapper.pre_check()
        finally:
            modules.globals.face_swap_model = original_model

        assert captured, "no calls captured"
        checksums = captured[0]["expected_checksums"] or {}
        checksum = checksums.get("inswapper_128_fp16.onnx", "")
        assert self._is_valid_sha256(checksum), (
            f"Checksum for inswapper_128_fp16.onnx is not a valid 64-char SHA-256 hex: {checksum!r}"
        )

    def test_gfpgan_checksum_is_valid_sha256_format(self):
        """Checksum for gfpgan-1024.onnx must be a valid SHA-256 hex string."""
        import modules.processors.frame.face_enhancer as face_enhancer

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append({"expected_checksums": expected_checksums})

        with (
            patch("modules.processors.frame.face_enhancer.conditional_download", side_effect=fake_download),
            patch("modules.processors.frame.face_enhancer.os.path.exists", return_value=False),
        ):
            face_enhancer.pre_check()

        assert captured, "no calls captured"
        checksums = captured[0]["expected_checksums"] or {}
        checksum = checksums.get("gfpgan-1024.onnx", "")
        assert self._is_valid_sha256(checksum), (
            f"Checksum for gfpgan-1024.onnx is not a valid 64-char SHA-256 hex: {checksum!r}"
        )

    def test_xseg_checksum_is_valid_sha256_format(self):
        """Checksum for xseg_2.onnx must be a valid SHA-256 hex string."""
        from modules import face_occluder

        captured: list[dict] = []

        def fake_download(directory, urls, expected_checksums=None):
            captured.append({"expected_checksums": expected_checksums})

        with (
            patch("modules.face_occluder.conditional_download", side_effect=fake_download),
            patch("modules.face_occluder.os.path.exists", return_value=False),
        ):
            face_occluder.pre_check()

        assert captured, "no calls captured"
        checksums = captured[0]["expected_checksums"] or {}
        checksum = checksums.get("xseg_2.onnx", "")
        assert self._is_valid_sha256(checksum), (
            f"Checksum for xseg_2.onnx is not a valid 64-char SHA-256 hex: {checksum!r}"
        )

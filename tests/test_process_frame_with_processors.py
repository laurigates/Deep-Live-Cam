"""Tests for _process_frame_with_processors helper (Issue #97).

Verifies that the extracted helper function exists, is callable with the
expected signature, and correctly routes each processor type (enhancer,
swapper, other).
"""

import inspect
import threading
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Existence and signature
# ---------------------------------------------------------------------------


class TestHelperExists:
    """The helper function must exist and be importable."""

    def test_helper_is_callable(self):
        from modules.ui_webcam import _process_frame_with_processors

        assert callable(_process_frame_with_processors)

    def test_helper_signature_has_required_params(self):
        from modules.ui_webcam import _process_frame_with_processors

        sig = inspect.signature(_process_frame_with_processors)
        params = set(sig.parameters.keys())
        required = {
            "frame_processor",
            "temp_frame",
            "map_faces",
            "skip_enhancer",
            "enhancement_seq",
            "last_consumed_enh_seq",
            "latest_enhanced_frame",
            "swap_seq",
            "last_consumed_swap_seq",
            "latest_swapped_frame",
            "enhancement_input",
            "enhancement_output",
            "enhancement_lock",
            "swap_input",
            "swap_output",
            "swap_lock",
        }
        missing = required - params
        assert not missing, f"Missing parameters: {missing}"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_frame(fill: int = 0) -> np.ndarray:
    return np.full((64, 64, 3), fill, dtype=np.uint8)


def _make_slots():
    """Return (input_holder, output_holder, lock) triple."""
    return [None], [None], threading.Lock()


def _call_helper(
    frame_processor,
    temp_frame,
    *,
    map_faces=False,
    skip_enhancer=False,
    enhancement_seq=0,
    last_consumed_enh_seq=-1,
    latest_enhanced_frame=None,
    swap_seq=0,
    last_consumed_swap_seq=-1,
    latest_swapped_frame=None,
    enhancement_input=None,
    enhancement_output=None,
    enhancement_lock=None,
    swap_input=None,
    swap_output=None,
    swap_lock=None,
    source_image=None,
    cached_target_face=None,
    cached_many_faces=None,
    cached_faces_list=None,
    prev_enhanced_faces=None,
):
    from modules.ui_webcam import _process_frame_with_processors

    if enhancement_input is None:
        enhancement_input, enhancement_output, enhancement_lock = _make_slots()
    if swap_input is None:
        swap_input, swap_output, swap_lock = _make_slots()
    return _process_frame_with_processors(
        frame_processor=frame_processor,
        temp_frame=temp_frame,
        map_faces=map_faces,
        skip_enhancer=skip_enhancer,
        enhancement_seq=enhancement_seq,
        last_consumed_enh_seq=last_consumed_enh_seq,
        latest_enhanced_frame=latest_enhanced_frame,
        swap_seq=swap_seq,
        last_consumed_swap_seq=last_consumed_swap_seq,
        latest_swapped_frame=latest_swapped_frame,
        enhancement_input=enhancement_input,
        enhancement_output=enhancement_output,
        enhancement_lock=enhancement_lock,
        swap_input=swap_input,
        swap_output=swap_output,
        swap_lock=swap_lock,
        source_image=source_image,
        cached_target_face=cached_target_face,
        cached_many_faces=cached_many_faces,
        cached_faces_list=cached_faces_list,
        prev_enhanced_faces=prev_enhanced_faces,
    )


# ---------------------------------------------------------------------------
# Swapper routing
# ---------------------------------------------------------------------------


class TestSwapperRouting:
    """When processor is DLC.FACE-SWAPPER, the helper submits to swap slot."""

    def test_swapper_submits_to_swap_input_normal_mode(self):
        swap_input, swap_output, swap_lock = _make_slots()
        frame = _make_frame(10)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-SWAPPER"

        result = _call_helper(
            processor,
            frame,
            map_faces=False,
            swap_seq=0,
            swap_input=swap_input,
            swap_output=swap_output,
            swap_lock=swap_lock,
        )

        # A submission should have been placed
        with swap_lock:
            inp = swap_input[0]
        assert inp is not None
        assert inp["map_faces"] is False
        assert inp["seq"] == 1  # seq was incremented from 0

    def test_swapper_submits_to_swap_input_map_faces_mode(self):
        swap_input, swap_output, swap_lock = _make_slots()
        frame = _make_frame(20)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-SWAPPER"

        result = _call_helper(
            processor,
            frame,
            map_faces=True,
            swap_seq=0,
            swap_input=swap_input,
            swap_output=swap_output,
            swap_lock=swap_lock,
        )

        with swap_lock:
            inp = swap_input[0]
        assert inp is not None
        assert inp["map_faces"] is True
        assert inp["source_face"] is None
        assert inp["target_face"] is None
        assert inp["many_faces"] is None

    def test_swapper_reads_swap_output_if_new_seq(self):
        swap_input, swap_output, swap_lock = _make_slots()
        swapped_frame = _make_frame(99)
        # Pre-populate output with seq=1
        swap_output[0] = {"frame": swapped_frame, "seq": 1}

        frame = _make_frame(10)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-SWAPPER"

        result = _call_helper(
            processor,
            frame,
            map_faces=False,
            swap_seq=0,  # will be incremented to 1, matching output
            last_consumed_swap_seq=-1,
            latest_swapped_frame=None,
            swap_input=swap_input,
            swap_output=swap_output,
            swap_lock=swap_lock,
        )

        # temp_frame should now be the swapped frame
        assert np.array_equal(result["temp_frame"], swapped_frame)
        assert result["latest_swapped_frame"] is not None


# ---------------------------------------------------------------------------
# Enhancer routing
# ---------------------------------------------------------------------------


class TestEnhancerRouting:
    """When processor is a known enhancer, the helper uses the enhancement slot."""

    def test_enhancer_not_submitted_when_skip(self):
        enh_input, enh_output, enh_lock = _make_slots()
        frame = _make_frame(30)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-ENHANCER"

        import modules.globals

        original_fp_ui = modules.globals.fp_ui.copy()
        modules.globals.fp_ui["face_enhancer"] = True
        try:
            result = _call_helper(
                processor,
                frame,
                map_faces=False,
                skip_enhancer=True,
                enhancement_input=enh_input,
                enhancement_output=enh_output,
                enhancement_lock=enh_lock,
            )
        finally:
            modules.globals.fp_ui.update(original_fp_ui)

        # skip_enhancer=True so nothing submitted
        with enh_lock:
            inp = enh_input[0]
        assert inp is None

    def test_enhancer_submitted_when_not_skip_and_enabled(self):
        enh_input, enh_output, enh_lock = _make_slots()
        frame = _make_frame(40)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-ENHANCER"

        import modules.globals

        original_fp_ui = modules.globals.fp_ui.copy()
        modules.globals.fp_ui["face_enhancer"] = True
        try:
            result = _call_helper(
                processor,
                frame,
                map_faces=False,
                skip_enhancer=False,
                enhancement_seq=0,
                enhancement_input=enh_input,
                enhancement_output=enh_output,
                enhancement_lock=enh_lock,
            )
        finally:
            modules.globals.fp_ui.update(original_fp_ui)

        with enh_lock:
            inp = enh_input[0]
        assert inp is not None
        assert inp["map_faces"] is False
        assert inp["seq"] == 1

    def test_enhancer_map_faces_submits_with_map_faces_true(self):
        enh_input, enh_output, enh_lock = _make_slots()
        frame = _make_frame(50)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-ENHANCER"

        import modules.globals

        original_fp_ui = modules.globals.fp_ui.copy()
        modules.globals.fp_ui["face_enhancer"] = True
        try:
            result = _call_helper(
                processor,
                frame,
                map_faces=True,
                skip_enhancer=False,
                enhancement_seq=0,
                enhancement_input=enh_input,
                enhancement_output=enh_output,
                enhancement_lock=enh_lock,
            )
        finally:
            modules.globals.fp_ui.update(original_fp_ui)

        with enh_lock:
            inp = enh_input[0]
        assert inp is not None
        assert inp["map_faces"] is True
        assert inp["faces"] is None

    def test_enhancer_disabled_clears_state(self):
        enh_input, enh_output, enh_lock = _make_slots()
        # Pre-populate input/output to verify they are cleared
        enh_input[0] = {"some": "data"}
        enh_output[0] = {"some": "data"}
        frame = _make_frame(60)
        processor = MagicMock()
        processor.NAME = "DLC.FACE-ENHANCER"

        import modules.globals

        original_fp_ui = modules.globals.fp_ui.copy()
        modules.globals.fp_ui["face_enhancer"] = False
        try:
            result = _call_helper(
                processor,
                frame,
                map_faces=False,
                latest_enhanced_frame=_make_frame(77),
                enhancement_input=enh_input,
                enhancement_output=enh_output,
                enhancement_lock=enh_lock,
            )
        finally:
            modules.globals.fp_ui.update(original_fp_ui)

        with enh_lock:
            assert enh_input[0] is None
            assert enh_output[0] is None
        assert result["latest_enhanced_frame"] is None


# ---------------------------------------------------------------------------
# Other processor fallthrough
# ---------------------------------------------------------------------------


class TestOtherProcessorFallthrough:
    """Processors that are neither enhancer nor swapper are called directly."""

    def test_other_processor_called_with_source_image_in_normal_mode(self):
        frame = _make_frame(70)
        source = _make_frame(80)
        processed = _make_frame(90)

        processor = MagicMock()
        processor.NAME = "DLC.FACE-MASKING"
        processor.process_frame.return_value = processed

        result = _call_helper(processor, frame, map_faces=False, source_image=source)

        processor.process_frame.assert_called_once_with(source, frame)
        assert np.array_equal(result["temp_frame"], processed)

    def test_other_processor_called_with_none_in_map_faces_mode(self):
        frame = _make_frame(70)
        processed = _make_frame(90)

        processor = MagicMock()
        processor.NAME = "DLC.FACE-MASKING"
        processor.process_frame.return_value = processed

        result = _call_helper(processor, frame, map_faces=True)

        processor.process_frame.assert_called_once_with(None, frame)
        assert np.array_equal(result["temp_frame"], processed)


# ---------------------------------------------------------------------------
# Return value contract
# ---------------------------------------------------------------------------


class TestReturnValueContract:
    """The helper must return a dict with the mutable state fields."""

    def test_returns_dict_with_required_keys(self):
        processor = MagicMock()
        processor.NAME = "DLC.FACE-MASKING"
        processor.process_frame.return_value = _make_frame(0)

        result = _call_helper(processor, _make_frame(0))

        required_keys = {
            "temp_frame",
            "enhancement_seq",
            "last_consumed_enh_seq",
            "latest_enhanced_frame",
            "swap_seq",
            "last_consumed_swap_seq",
            "latest_swapped_frame",
            "prev_enhanced_faces",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in return value: {missing}"

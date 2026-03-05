"""Tests for modules/iobinding.py — IOBinding utility for CUDA GPU buffer reuse."""

from unittest.mock import MagicMock, patch

import numpy as np


def _mock_session(
    provider: str,
    input_name: str = "input",
    output_name: str = "output",
    output_shape: list | None = None,
    output_type: str = "tensor(float)",
):
    """Create a mock ONNX InferenceSession with the given provider."""
    session = MagicMock()
    session.get_providers.return_value = [provider]

    mock_input = MagicMock()
    mock_input.name = input_name
    mock_input.shape = [1, 3, 512, 512]
    mock_input.type = "tensor(float)"
    session.get_inputs.return_value = [mock_input]

    mock_output = MagicMock()
    mock_output.name = output_name
    mock_output.shape = output_shape or [1, 3, 1024, 1024]
    mock_output.type = output_type
    session.get_outputs.return_value = [mock_output]

    return session


class TestSupportsIOBinding:
    """Tests for supports_iobinding() provider detection."""

    def test_cuda_provider_returns_true(self):
        session = _mock_session("CUDAExecutionProvider")
        from modules.iobinding import supports_iobinding

        assert supports_iobinding(session) is True

    def test_coreml_provider_returns_false(self):
        session = _mock_session("CoreMLExecutionProvider")
        from modules.iobinding import supports_iobinding

        assert supports_iobinding(session) is False

    def test_cpu_provider_returns_false(self):
        session = _mock_session("CPUExecutionProvider")
        from modules.iobinding import supports_iobinding

        assert supports_iobinding(session) is False


class TestCreateIOBindingContext:
    """Tests for create_iobinding_context() factory function."""

    def test_returns_none_for_cpu(self):
        session = _mock_session("CPUExecutionProvider")
        from modules.iobinding import create_iobinding_context

        assert create_iobinding_context(session) is None

    def test_returns_none_on_error(self):
        session = _mock_session("CUDAExecutionProvider")
        session.io_binding.side_effect = RuntimeError("IOBinding not supported")
        from modules.iobinding import create_iobinding_context

        result = create_iobinding_context(session)
        assert result is None

    def test_returns_context_for_cuda(self):
        session = _mock_session("CUDAExecutionProvider")
        # io_binding() returns a mock binding object
        session.io_binding.return_value = MagicMock()
        from modules.iobinding import IOBindingContext, create_iobinding_context

        with patch("modules.iobinding.onnxruntime") as mock_ort:
            mock_ort.OrtValue.ortvalue_from_shape_and_type.return_value = MagicMock()
            ctx = create_iobinding_context(session)
            assert ctx is not None
            assert isinstance(ctx, IOBindingContext)


class TestIOBindingContextRun:
    """Tests for IOBindingContext.run() method."""

    def test_run_calls_run_with_iobinding(self):
        session = _mock_session("CUDAExecutionProvider")
        mock_binding = MagicMock()
        session.io_binding.return_value = mock_binding

        from modules.iobinding import IOBindingContext

        with patch("modules.iobinding.onnxruntime") as mock_ort:
            mock_ort_value = MagicMock()
            mock_ort_value.numpy.return_value = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
            mock_ort.OrtValue.ortvalue_from_shape_and_type.return_value = mock_ort_value

            ctx = IOBindingContext(session)
            input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)
            ctx.run({"input": input_tensor})

            session.run_with_iobinding.assert_called_once_with(mock_binding)

    def test_run_returns_output_list(self):
        session = _mock_session("CUDAExecutionProvider")
        mock_binding = MagicMock()
        session.io_binding.return_value = mock_binding

        from modules.iobinding import IOBindingContext

        with patch("modules.iobinding.onnxruntime") as mock_ort:
            expected = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
            mock_ort_value = MagicMock()
            mock_ort_value.numpy.return_value = expected
            mock_ort.OrtValue.ortvalue_from_shape_and_type.return_value = mock_ort_value

            ctx = IOBindingContext(session)
            result = ctx.run({"input": np.zeros((1, 3, 512, 512), dtype=np.float32)})

            assert isinstance(result, list)
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], expected)

    def test_preallocates_output_buffers(self):
        """IOBindingContext should pre-allocate GPU output OrtValues from session metadata."""
        session = _mock_session("CUDAExecutionProvider", output_shape=[1, 3, 1024, 1024], output_type="tensor(float)")
        session.io_binding.return_value = MagicMock()

        from modules.iobinding import IOBindingContext

        with patch("modules.iobinding.onnxruntime") as mock_ort:
            mock_ort.OrtValue.ortvalue_from_shape_and_type.return_value = MagicMock()
            IOBindingContext(session)

            # Should have called ortvalue_from_shape_and_type for the output
            mock_ort.OrtValue.ortvalue_from_shape_and_type.assert_called_once_with(
                [1, 3, 1024, 1024], np.float32, "cuda", 0
            )

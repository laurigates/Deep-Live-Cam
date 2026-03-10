"""Tests for download_model_if_needed in modules/utilities.py.

Updated in Issue #61: verifies the on_status injectable callback parameter.
"""

from unittest.mock import MagicMock, patch


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_returns_true_when_model_exists(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = True
    result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST", models_dir="/tmp/models")
    assert result is True
    mock_download.assert_called_once_with("/tmp/models", ["http://example.com/model.onnx"])


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_returns_false_when_model_missing_after_download(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = False
    result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST", models_dir="/tmp/models")
    assert result is False


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_uses_default_models_dir(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = True
    result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST")
    assert result is True
    call_args = mock_download.call_args[0]
    assert call_args[0].endswith("models")


# ---------------------------------------------------------------------------
# Injectable on_status callback (#61)
# ---------------------------------------------------------------------------


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_on_status_called_when_downloading(mock_exists, mock_download):
    """When on_status is provided and model is missing, it is called before download."""
    from modules.utilities import download_model_if_needed

    mock_exists.side_effect = [False, True]  # missing → found after download
    status_calls = []

    def on_status(msg, name):
        status_calls.append((msg, name))

    result = download_model_if_needed(
        "model.onnx",
        ["http://example.com/model.onnx"],
        "TEST",
        models_dir="/tmp/models",
        on_status=on_status,
    )
    assert result is True
    assert len(status_calls) == 1
    assert status_calls[0] == ("Downloading model.onnx...", "TEST")


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_on_status_called_on_download_failure(mock_exists, mock_download):
    """When download fails (model still missing), on_status is called with failure message."""
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = False
    status_calls = []

    def on_status(msg, name):
        status_calls.append((msg, name))

    result = download_model_if_needed(
        "model.onnx",
        ["http://example.com/model.onnx"],
        "PROC",
        models_dir="/tmp/models",
        on_status=on_status,
    )
    assert result is False
    # Two status calls: one for "Downloading..." and one for "Model not found..."
    assert len(status_calls) == 2
    assert status_calls[0][0] == "Downloading model.onnx..."
    assert "not found" in status_calls[1][0].lower() or "failed" in status_calls[1][0].lower()


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_no_core_import_when_on_status_provided(mock_exists, mock_download):
    """When on_status callback is provided, modules.core must not be imported."""
    import sys

    from modules.utilities import download_model_if_needed

    mock_exists.side_effect = [False, True]
    on_status = MagicMock()

    # Ensure modules.core is not in sys.modules so import would be detectable
    core_was_loaded = "modules.core" in sys.modules
    if core_was_loaded:
        original_core = sys.modules.pop("modules.core")

    try:
        download_model_if_needed(
            "model.onnx",
            ["http://example.com/model.onnx"],
            "TEST",
            models_dir="/tmp/models",
            on_status=on_status,
        )
        # If on_status was used, modules.core should not have been imported
        if not core_was_loaded:
            assert "modules.core" not in sys.modules, "modules.core was imported despite on_status being provided"
    finally:
        if core_was_loaded:
            sys.modules["modules.core"] = original_core

    on_status.assert_called()


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_on_status_not_required(mock_exists, mock_download):
    """Omitting on_status is valid — no error raised."""
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = True
    # Should not raise even with no on_status
    result = download_model_if_needed("model.onnx", ["http://x.com"], "T", models_dir="/tmp")
    assert result is True


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_on_status_signature_is_accepted(mock_exists, mock_download):
    """Verify on_status parameter exists in the function signature."""
    import inspect

    from modules.utilities import download_model_if_needed

    sig = inspect.signature(download_model_if_needed)
    assert "on_status" in sig.parameters

"""Tests for --lang CLI argument sanitization in LanguageManager.load_language.

Verifies that path traversal attacks and invalid inputs are rejected before
any filesystem path is constructed.
"""

import pytest
from modules.gettext import LanguageManager


@pytest.fixture
def manager():
    return LanguageManager(default_language="en")


class TestPathTraversalRejection:
    def test_dotdot_slash_is_rejected(self, manager):
        result = manager.load_language("../../etc/passwd")
        assert result is False

    def test_dotdot_backslash_is_rejected(self, manager):
        result = manager.load_language("..\\..\\windows\\system32")
        assert result is False

    def test_leading_slash_is_rejected(self, manager):
        result = manager.load_language("/etc/passwd")
        assert result is False

    def test_embedded_slash_is_rejected(self, manager):
        result = manager.load_language("en/../../etc/passwd")
        assert result is False

    def test_null_byte_is_rejected(self, manager):
        result = manager.load_language("en\x00evil")
        assert result is False


class TestEmptyAndLengthGuards:
    def test_empty_string_is_rejected(self, manager):
        result = manager.load_language("")
        assert result is False

    def test_overly_long_code_is_rejected(self, manager):
        result = manager.load_language("a" * 11)
        assert result is False

    def test_max_length_boundary_is_accepted_or_not_found(self, manager):
        # A 10-char alphanumeric code is valid syntax (may or may not have a locale file)
        result = manager.load_language("a" * 10)
        # Must not raise; False means not found (expected), True means found
        assert isinstance(result, bool)


class TestSpecialCharacterRejection:
    def test_dot_in_code_is_rejected(self, manager):
        result = manager.load_language("en.malicious")
        assert result is False

    def test_semicolon_is_rejected(self, manager):
        result = manager.load_language("en;rm -rf /")
        assert result is False

    def test_percent_encoding_is_rejected(self, manager):
        result = manager.load_language("%2e%2e%2fetc%2fpasswd")
        assert result is False

    def test_space_is_rejected(self, manager):
        result = manager.load_language("en us")
        assert result is False


class TestValidCodesAreAccepted:
    """Valid codes must not raise; return value depends on whether the locale file exists."""

    def test_zh_does_not_raise(self, manager):
        result = manager.load_language("zh")
        assert isinstance(result, bool)

    def test_ja_does_not_raise(self, manager):
        result = manager.load_language("ja")
        assert isinstance(result, bool)

    def test_de_does_not_raise(self, manager):
        result = manager.load_language("de")
        assert isinstance(result, bool)

    def test_ko_does_not_raise(self, manager):
        result = manager.load_language("ko")
        assert isinstance(result, bool)

    def test_pt_br_hyphenated_code_does_not_raise(self, manager):
        # pt-br is a real locale shipped with the project; hyphens must be allowed
        result = manager.load_language("pt-br")
        assert isinstance(result, bool)

    def test_en_short_circuit_returns_true(self, manager):
        result = manager.load_language("en")
        assert result is True


class TestResolvedPathStaysInLocalesDir:
    """Ensure resolved path cannot escape the locales directory even with exotic inputs."""

    def test_path_traversal_with_unicode_separators_is_rejected(self, manager):
        # Unicode look-alike separators that some systems normalise to /
        result = manager.load_language("en\u2215etc\u2215passwd")
        assert result is False

    def test_windows_style_absolute_path_is_rejected(self, manager):
        result = manager.load_language("C:\\Windows\\system32")
        assert result is False

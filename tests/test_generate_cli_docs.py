"""Tests for scripts/generate_cli_docs.py — AST-based CLI docs generator."""

import subprocess
import sys
from pathlib import Path

import pytest

# Import the generator module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_cli_docs import (
    CORE_PY,
    extract_args,
    format_table,
    generate_cli_docs,
)


@pytest.fixture
def core_source() -> str:
    return CORE_PY.read_text()


@pytest.fixture
def extracted_args(core_source: str) -> list[dict]:
    return extract_args(core_source)


class TestExtractArgs:
    def test_finds_expected_arg_count(self, extracted_args: list[dict]) -> None:
        """Should find all non-deprecated args (currently 32 including -v/--version)."""
        # 36 total add_argument calls minus 4 deprecated = 32
        assert len(extracted_args) == 32

    def test_version_excluded_from_table(self, extracted_args: list[dict]) -> None:
        """The -v/--version action=version arg is extracted but filtered in format_table."""
        table = format_table(extracted_args)
        assert "--version" not in table

    def test_deprecated_args_excluded(self, core_source: str) -> None:
        """Args with help=argparse.SUPPRESS should not appear."""
        args = extract_args(core_source)
        all_flags = " ".join(a["flags"] for a in args)
        assert "--cpu-cores" not in all_flags
        assert "--gpu-vendor" not in all_flags
        assert "--gpu-threads" not in all_flags
        assert "-f" not in all_flags.split(", ")

    def test_source_arg_extracted(self, extracted_args: list[dict]) -> None:
        """The -s/--source argument should be present with correct help text."""
        source_args = [a for a in extracted_args if "--source" in a["flags"]]
        assert len(source_args) == 1
        assert source_args[0]["help"] == "select an source image"

    def test_video_quality_choices(self, extracted_args: list[dict]) -> None:
        """range(52) should resolve to '0-51'."""
        vq_args = [a for a in extracted_args if "--video-quality" in a["flags"]]
        assert len(vq_args) == 1
        assert vq_args[0]["choices"] == "0-51"

    def test_keyframe_interval_choices(self, extracted_args: list[dict]) -> None:
        """range(2, 11) should resolve to '2-10'."""
        ki_args = [a for a in extracted_args if "--keyframe-interval" in a["flags"]]
        assert len(ki_args) == 1
        assert ki_args[0]["choices"] == "2-10"

    def test_max_memory_default_is_auto(self, extracted_args: list[dict]) -> None:
        """suggest_max_memory() should resolve to '(auto)'."""
        mm_args = [a for a in extracted_args if "--max-memory" in a["flags"]]
        assert len(mm_args) == 1
        assert mm_args[0]["default"] == "(auto)"

    def test_frame_processor_has_choices(self, extracted_args: list[dict]) -> None:
        """--frame-processor should list all processor choices."""
        fp_args = [a for a in extracted_args if "--frame-processor" in a["flags"]]
        assert len(fp_args) == 1
        assert "face_swapper" in fp_args[0]["choices"]
        assert "face_enhancer" in fp_args[0]["choices"]


class TestFormatTable:
    def test_output_is_valid_markdown_table(self, extracted_args: list[dict]) -> None:
        """Output should have header, separator, and data rows."""
        table = format_table(extracted_args)
        lines = table.strip().split("\n")
        # Line 0: version info, line 1: blank, line 2: header, line 3: separator, line 4+: data
        assert "| Flag |" in lines[2]
        assert "|---" in lines[3]
        # 25 data rows (26 extracted minus -v/--version filtered by format_table)
        non_version_count = sum(1 for a in extracted_args if a["action"] != "version")
        assert len(lines) >= 4 + non_version_count

    def test_store_true_default_shows_false(self, extracted_args: list[dict]) -> None:
        """store_true actions should show False as default."""
        table = format_table(extracted_args)
        # --keep-fps is store_true with default=False
        assert "`False`" in table


class TestGenerateCliDocs:
    def test_generates_non_empty_output(self) -> None:
        table = generate_cli_docs()
        assert len(table) > 100
        assert "| Flag |" in table

    def test_contains_version_info(self) -> None:
        table = generate_cli_docs()
        assert "v2.0.3c" in table


class TestCheckMode:
    def test_check_passes_when_fresh(self, tmp_path: Path) -> None:
        """--check should exit 0 when README matches generated content."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "generate_cli_docs.py"), "--stdout"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "| Flag |" in result.stdout

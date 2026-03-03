"""
Verify that git-sourced dependencies in pyproject.toml use commit hash pinning
rather than mutable branch references like @master or @main.
"""
import re
import pathlib


PYPROJECT_PATH = pathlib.Path(__file__).parent.parent / "pyproject.toml"
# Full 40-char hex SHA pattern
_SHA_RE = re.compile(r"git\+https://[^@]+@([0-9a-f]{40})\b")
# Mutable branch references that must not appear in git URLs
_MUTABLE_REF_RE = re.compile(r"git\+https://[^@]+@(master|main)\b")


def _read_pyproject() -> str:
    return PYPROJECT_PATH.read_text(encoding="utf-8")


def test_no_master_branch_in_git_deps():
    """No git dependency may point at the @master branch."""
    content = _read_pyproject()
    matches = _MUTABLE_REF_RE.findall(content)
    assert matches == [], (
        f"Found mutable branch reference(s) in git dependencies: {matches}. "
        "Replace @master/@main with a full commit SHA."
    )


def test_no_main_branch_in_git_deps():
    """No git dependency may point at the @main branch."""
    content = _read_pyproject()
    # _MUTABLE_REF_RE already covers 'main', but be explicit for clarity
    main_matches = re.findall(r"git\+https://[^@]+@main\b", content)
    assert main_matches == [], (
        f"Found @main reference(s) in git dependencies: {main_matches}. "
        "Replace with a full commit SHA."
    )


def test_basicsr_pinned_to_commit_hash():
    """basicsr git dependency is pinned to a full 40-char commit SHA."""
    content = _read_pyproject()
    line = next(
        (ln for ln in content.splitlines() if "BasicSR" in ln and "git+" in ln),
        None,
    )
    assert line is not None, "basicsr git dependency not found in pyproject.toml"
    assert _SHA_RE.search(line), (
        f"basicsr is not pinned to a commit hash. Found: {line.strip()}"
    )


def test_gfpgan_pinned_to_commit_hash():
    """gfpgan git dependency is pinned to a full 40-char commit SHA."""
    content = _read_pyproject()
    line = next(
        (ln for ln in content.splitlines() if "GFPGAN" in ln and "git+" in ln),
        None,
    )
    assert line is not None, "gfpgan git dependency not found in pyproject.toml"
    assert _SHA_RE.search(line), (
        f"gfpgan is not pinned to a commit hash. Found: {line.strip()}"
    )

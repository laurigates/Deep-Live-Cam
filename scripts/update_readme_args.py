#!/usr/bin/env python3
"""Update the CLI arguments section in README.md from `run.py --help` output.

Usage:
    uv run scripts/update_readme_args.py

The script captures `uv run run.py --help`, then replaces everything between
the marker comments in README.md:

    <!-- CLI_ARGS_START -->
    ...
    <!-- CLI_ARGS_END -->
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
START_MARKER = "<!-- CLI_ARGS_START -->"
END_MARKER = "<!-- CLI_ARGS_END -->"


def get_help_text() -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"run.py --help failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def update_readme(help_text: str) -> None:
    content = README.read_text()

    if START_MARKER not in content or END_MARKER not in content:
        print(
            f"Markers {START_MARKER} / {END_MARKER} not found in README.md",
            file=sys.stderr,
        )
        sys.exit(1)

    before = content[: content.index(START_MARKER) + len(START_MARKER)]
    after = content[content.index(END_MARKER) :]

    new_section = f"\n```\n{help_text}\n```\n"
    README.write_text(before + new_section + after)
    print("README.md updated.")


if __name__ == "__main__":
    help_text = get_help_text()
    update_readme(help_text)

"""Tests that verify version bounds are set on critical dependencies in pyproject.toml."""

import re
from pathlib import Path

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def _get_dependencies() -> list[str]:
    content = PYPROJECT_PATH.read_text()
    # Extract the dependencies list between the brackets
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    assert match, "Could not find dependencies section in pyproject.toml"
    deps_block = match.group(1)
    # Extract individual quoted strings
    return re.findall(r'"([^"]+)"', deps_block)


def test_numpy_has_upper_bound():
    deps = _get_dependencies()
    numpy_deps = [d for d in deps if d.startswith("numpy")]
    assert numpy_deps, "numpy dependency not found in pyproject.toml"
    numpy_dep = numpy_deps[0]
    assert "<" in numpy_dep, (
        f"numpy dependency '{numpy_dep}' is missing an upper bound (<). "
        "Add an upper bound to prevent incompatible future versions from being installed."
    )


def test_torch_darwin_has_lower_bound():
    deps = _get_dependencies()
    # Find the torch entry for darwin (no sys_platform != 'darwin')
    torch_darwin = [
        d
        for d in deps
        if d.startswith("torch")
        and "sys_platform == 'darwin'" in d
        and "!=" not in d
        and not d.startswith("torchvision")
    ]
    assert torch_darwin, "torch darwin dependency not found in pyproject.toml"
    dep = torch_darwin[0]
    assert ">=" in dep or ">" in dep, (
        f"torch darwin dependency '{dep}' is missing a lower bound (>= or >). "
        "Add a lower bound to ensure a minimum compatible version."
    )


def test_torch_darwin_has_upper_bound():
    deps = _get_dependencies()
    torch_darwin = [
        d
        for d in deps
        if d.startswith("torch")
        and "sys_platform == 'darwin'" in d
        and "!=" not in d
        and not d.startswith("torchvision")
    ]
    assert torch_darwin, "torch darwin dependency not found in pyproject.toml"
    dep = torch_darwin[0]
    assert "<" in dep, (
        f"torch darwin dependency '{dep}' is missing an upper bound (<). "
        "Add an upper bound to prevent incompatible future major versions."
    )

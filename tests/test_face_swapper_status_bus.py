"""Tests verifying face_swapper uses StatusBus instead of modules.core.update_status.

Issue #99: Replace core.update_status import in face_swapper with StatusBus.
"""

import ast
from pathlib import Path

_FACE_SWAPPER_PATH = Path(__file__).parent.parent / "modules" / "processors" / "frame" / "face_swapper.py"
_FACE_OCCLUDER_PATH = Path(__file__).parent.parent / "modules" / "face_occluder.py"


def _parse_imports(source_path: Path) -> list[tuple[str, list[str]]]:
    """Return list of (module, names) for all ImportFrom nodes in the file."""
    tree = ast.parse(source_path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            imports.append((node.module, names))
    return imports


class TestFaceSwapperDoesNotImportFromCore:
    """face_swapper.py must not import update_status from modules.core."""

    def test_no_from_modules_core_import(self):
        imports = _parse_imports(_FACE_SWAPPER_PATH)
        core_imports = [names for mod, names in imports if mod == "modules.core"]
        flat = [name for names in core_imports for name in names]
        assert "update_status" not in flat, (
            "face_swapper.py still imports update_status from modules.core. "
            "Replace with BUS.publish from modules.status_bus."
        )

    def test_imports_status_bus(self):
        imports = _parse_imports(_FACE_SWAPPER_PATH)
        bus_imports = [names for mod, names in imports if mod == "modules.status_bus"]
        flat = [name for names in bus_imports for name in names]
        assert "BUS" in flat, "face_swapper.py does not import BUS from modules.status_bus."

    def test_no_bare_update_status_calls(self):
        source = _FACE_SWAPPER_PATH.read_text()
        # After the fix, there should be no standalone update_status( calls
        import re

        # Match update_status( that is NOT preceded by "." (i.e., not a method call)
        pattern = re.compile(r"(?<!\.)update_status\s*\(")
        matches = pattern.findall(source)
        assert not matches, (
            f"face_swapper.py still contains {len(matches)} bare update_status() call(s). "
            "Replace all with BUS.publish()."
        )


class TestFaceOccluderDoesNotImportFromCore:
    """face_occluder.py must not import update_status from modules.core."""

    def test_no_from_modules_core_import(self):
        imports = _parse_imports(_FACE_OCCLUDER_PATH)
        core_imports = [names for mod, names in imports if mod == "modules.core"]
        flat = [name for names in core_imports for name in names]
        assert "update_status" not in flat, (
            "face_occluder.py still imports update_status from modules.core. "
            "Replace with BUS.publish from modules.status_bus."
        )

    def test_imports_status_bus(self):
        imports = _parse_imports(_FACE_OCCLUDER_PATH)
        bus_imports = [names for mod, names in imports if mod == "modules.status_bus"]
        flat = [name for names in bus_imports for name in names]
        assert "BUS" in flat, "face_occluder.py does not import BUS from modules.status_bus."

    def test_no_bare_update_status_calls(self):
        source = _FACE_OCCLUDER_PATH.read_text()
        import re

        pattern = re.compile(r"(?<!\.)update_status\s*\(")
        matches = pattern.findall(source)
        assert not matches, (
            f"face_occluder.py still contains {len(matches)} bare update_status() call(s). "
            "Replace all with BUS.publish()."
        )

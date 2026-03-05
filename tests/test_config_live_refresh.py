"""Tests verifying ProcessingConfig is rebuilt each frame in the webcam loop.

The processing thread in ui_webcam._processing_thread_func must call
build_config_from_globals() inside its while loop so that UI toggle changes
(occlusion_mask, mouth_mask, opacity, etc.) propagate immediately to
downstream processors.
"""

import ast

import modules.globals
from modules.processing_config_factory import build_config_from_globals

# ---------------------------------------------------------------------------
# AST-based structural tests
# ---------------------------------------------------------------------------


class TestConfigRefreshInsideWhileLoop:
    """Verify build_config_from_globals() is called inside the while loop."""

    @staticmethod
    def _get_processing_thread_func_ast():
        """Parse ui_webcam.py and return the AST of _processing_thread_func."""
        import pathlib

        src = pathlib.Path(__file__).resolve().parent.parent / "modules" / "ui_webcam.py"
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_processing_thread_func":
                return node
        raise AssertionError("_processing_thread_func not found in ui_webcam.py")

    @staticmethod
    def _find_while_loops(func_node):
        """Return all While nodes that are direct children of the function body."""
        return [n for n in ast.walk(func_node) if isinstance(n, ast.While)]

    def test_build_config_called_inside_while_loop(self):
        func = self._get_processing_thread_func_ast()
        while_loops = self._find_while_loops(func)
        assert while_loops, "No while loop found in _processing_thread_func"

        # Look for `config = build_config_from_globals()` inside any while loop
        found = False
        for loop in while_loops:
            for node in ast.walk(loop):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "config"
                    and isinstance(node.value, ast.Call)
                ):
                    call = node.value
                    func_name = ""
                    if isinstance(call.func, ast.Name):
                        func_name = call.func.id
                    elif isinstance(call.func, ast.Attribute):
                        func_name = call.func.attr
                    if func_name == "build_config_from_globals":
                        found = True
                        break
            if found:
                break

        assert found, "Expected `config = build_config_from_globals()` inside the while loop of _processing_thread_func"

    def test_build_config_call_precedes_snapshot_section(self):
        """build_config_from_globals() must appear before the snapshot assignments."""
        func = self._get_processing_thread_func_ast()
        while_loops = self._find_while_loops(func)
        assert while_loops

        for loop in while_loops:
            config_line = None
            first_snap_line = None
            for node in ast.walk(loop):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name == "config" and isinstance(node.value, ast.Call):
                        call = node.value
                        fname = ""
                        if isinstance(call.func, ast.Name):
                            fname = call.func.id
                        elif isinstance(call.func, ast.Attribute):
                            fname = call.func.attr
                        if fname == "build_config_from_globals":
                            config_line = node.lineno
                    elif name.startswith("snap_") and first_snap_line is None:
                        first_snap_line = node.lineno

            if config_line is not None and first_snap_line is not None:
                assert config_line < first_snap_line, (
                    f"build_config_from_globals (line {config_line}) must appear "
                    f"before first snap_ assignment (line {first_snap_line})"
                )
                return

        raise AssertionError("Could not find both config assignment and snap_ assignments in while loop")


# ---------------------------------------------------------------------------
# Functional tests — verify toggle changes propagate through config rebuild
# ---------------------------------------------------------------------------


class TestConfigRefreshPicksUpToggles:
    """Verify that toggling globals is reflected in rebuilt configs."""

    def test_occlusion_mask_toggle(self):
        original = modules.globals.occlusion_mask
        try:
            modules.globals.occlusion_mask = False
            config1 = build_config_from_globals()
            assert config1.occlusion_mask is False

            modules.globals.occlusion_mask = True
            config2 = build_config_from_globals()
            assert config2.occlusion_mask is True
        finally:
            modules.globals.occlusion_mask = original

    def test_mouth_mask_toggle(self):
        original = modules.globals.mouth_mask
        try:
            modules.globals.mouth_mask = False
            config1 = build_config_from_globals()
            assert config1.mouth_mask is False

            modules.globals.mouth_mask = True
            config2 = build_config_from_globals()
            assert config2.mouth_mask is True
        finally:
            modules.globals.mouth_mask = original

    def test_opacity_change(self):
        original = getattr(modules.globals, "opacity", 1.0)
        try:
            modules.globals.opacity = 0.5
            config1 = build_config_from_globals()
            assert config1.opacity == 0.5

            modules.globals.opacity = 0.8
            config2 = build_config_from_globals()
            assert config2.opacity == 0.8
        finally:
            modules.globals.opacity = original

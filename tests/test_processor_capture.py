"""
AST-based test verifying that `get_frame_processors_modules` is called once
before the while loop in `_processing_thread_func`, not inside it.

Issue #93: capturing processors per-frame caused ~30 unnecessary lock
acquisitions per second at 30 FPS.
"""

import ast
import pathlib

_WEBCAM_FILE = pathlib.Path(__file__).parent.parent / "modules" / "ui_webcam.py"


def _parse_func(source: str, func_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise ValueError(f"Function {func_name!r} not found in source")


def _find_call_lines(tree: ast.AST, func_name: str) -> list[int]:
    """Return line numbers of all calls to `func_name` anywhere in `tree`."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
            lines.append(node.lineno)
    return lines


def _while_loop_line_range(func_node: ast.FunctionDef) -> tuple[int, int]:
    """Return (start, end) line numbers of the first while loop in the function."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.While):
            return node.lineno, node.end_lineno
    raise ValueError("No while loop found in function")


class TestProcessorCaptureBeforeLoop:
    def test_get_frame_processors_modules_called_before_while_loop(self):
        """get_frame_processors_modules must be called before the while loop."""
        source = _WEBCAM_FILE.read_text()
        func = _parse_func(source, "_processing_thread_func")

        while_start, while_end = _while_loop_line_range(func)

        # Find all call sites of get_frame_processors_modules inside the function
        call_lines = _find_call_lines(func, "get_frame_processors_modules")

        assert call_lines, "get_frame_processors_modules is never called in _processing_thread_func"

        calls_inside_loop = [ln for ln in call_lines if while_start <= ln <= while_end]
        calls_before_loop = [ln for ln in call_lines if ln < while_start]

        assert not calls_inside_loop, (
            f"get_frame_processors_modules is called inside the while loop "
            f"(line(s) {calls_inside_loop}). Move it before the loop (line {while_start})."
        )
        assert calls_before_loop, "get_frame_processors_modules must be called before the while loop."

    def test_get_frame_processors_modules_called_exactly_once(self):
        """get_frame_processors_modules should appear exactly once in _processing_thread_func."""
        source = _WEBCAM_FILE.read_text()
        func = _parse_func(source, "_processing_thread_func")
        call_lines = _find_call_lines(func, "get_frame_processors_modules")

        assert len(call_lines) == 1, (
            f"Expected exactly 1 call to get_frame_processors_modules, found {len(call_lines)} at lines {call_lines}."
        )

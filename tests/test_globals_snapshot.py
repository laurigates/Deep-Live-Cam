"""
Tests that _processing_thread_func snapshots all mutable globals at the top of the
while loop before using them in frame processing.

This is a source-code inspection test — it parses the AST of ui_webcam.py and verifies
that every expected `snap_X` variable is assigned from `modules.globals.X` before any
read of `modules.globals.X` occurs inside the while loop body.
"""

import ast
import inspect
import textwrap

import pytest

import modules.ui_webcam as ui_webcam

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_func_source(func) -> str:
    """Return dedented source of a function."""
    return textwrap.dedent(inspect.getsource(func))


def _find_while_loop_node(func_source: str) -> ast.While:
    """Return the first While node in the function body."""
    tree = ast.parse(func_source)
    func_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    for node in func_def.body:
        if isinstance(node, ast.While):
            return node
    raise AssertionError("No while loop found in _processing_thread_func")


def _collect_assignments_at_loop_top(while_node: ast.While) -> dict[str, str]:
    """
    Collect all simple assignments at the *top* of the while body before any
    if/for/with block.  Returns mapping snap_name -> globals attribute name.

    We stop collecting when we hit a non-assignment statement (other than the
    try/except that pops from capture_queue, which always comes first).
    """
    assignments: dict[str, str] = {}
    # The first statement is the try/except for capture_queue.get — skip it
    # and collect assignments until the first compound statement.
    for stmt in while_node.body:
        if isinstance(stmt, ast.Try):
            # This is the capture_queue.get block — skip it
            continue
        if not isinstance(stmt, ast.Assign):
            # First non-assignment after the try block: snapshot section ends
            break
        # Check that the RHS is modules.globals.<attr>
        target = stmt.targets
        if len(target) != 1:
            continue
        t = target[0]
        if not isinstance(t, ast.Name):
            continue
        val = stmt.value
        # Accept plain attribute reads: modules.globals.X
        if (
            isinstance(val, ast.Attribute)
            and isinstance(val.value, ast.Attribute)
            and isinstance(val.value.value, ast.Name)
            and val.value.value.id == "modules"
            and val.value.attr == "globals"
        ):
            assignments[t.id] = val.attr
        # Accept max(N, modules.globals.X)
        elif isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "max":
            for arg in val.args:
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Attribute)
                    and isinstance(arg.value.value, ast.Name)
                    and arg.value.value.id == "modules"
                    and arg.value.attr == "globals"
                ):
                    assignments[t.id] = arg.attr
    return assignments


def _collect_globals_reads_after_snapshot(while_node: ast.While) -> set[str]:
    """
    Walk the while body *after* the snapshot section and collect every
    modules.globals.<attr> read.  The snapshot section is: the leading
    try/except (capture_queue.get) + all consecutive simple assignments.
    Returns the set of attribute names read in the rest of the body.
    """
    reads: set[str] = set()

    # Find where the snapshot section ends (first compound/non-assign stmt after try)
    post_snapshot_stmts = []
    past_try = False
    for stmt in while_node.body:
        if not past_try:
            if isinstance(stmt, ast.Try):
                past_try = True
            continue
        if isinstance(stmt, ast.Assign):
            # Still in snapshot section — skip
            continue
        # First non-assignment after the try: rest of loop body
        post_snapshot_stmts.append(stmt)

    # Collect all modules.globals reads in the post-snapshot statements
    for stmt in post_snapshot_stmts:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "modules"
                and node.value.attr == "globals"
            ):
                if not isinstance(node.ctx, ast.Store):
                    reads.add(node.attr)
    return reads


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_SNAP_VARS = {
    # snap_var_name: globals_attr_name
    "snap_live_mirror": "live_mirror",
    "snap_half_rate": "half_rate_processing",
    "snap_keyframe_interval": "keyframe_interval",
    "snap_map_faces": "map_faces",
    "snap_source_path": "source_path",
    "snap_many_faces": "many_faces",
    "snap_motion_adaptive": "motion_adaptive_enhancement",
    "snap_iou_thresh": "motion_adaptive_iou_threshold",
    "snap_cos_thresh": "motion_adaptive_cosine_threshold",
    "snap_enh_interval": "enhancer_skip_interval",
    "snap_rife_enabled": "rife_enabled",
    "snap_rife_multiplier": "rife_multiplier",
    "snap_virtual_cam": "virtual_cam",
}


class TestGlobalsSnapshot:
    """Verify that _processing_thread_func snapshots globals at the while-loop top."""

    @pytest.fixture(scope="class")
    def while_node(self):
        src = _get_func_source(ui_webcam._processing_thread_func)
        return _find_while_loop_node(src)

    def test_snapshot_vars_present_at_loop_top(self, while_node):
        """All expected snap_X variables must be assigned at the top of the while loop."""
        assignments = _collect_assignments_at_loop_top(while_node)
        missing = {k for k in EXPECTED_SNAP_VARS if k not in assignments}
        assert not missing, (
            f"Missing snapshot assignments at top of while loop: {sorted(missing)}\n"
            f"Found assignments: {sorted(assignments.keys())}"
        )

    def test_snapshot_vars_assigned_from_correct_globals(self, while_node):
        """Each snap_X must be assigned from the matching modules.globals attribute."""
        assignments = _collect_assignments_at_loop_top(while_node)
        wrong = {}
        for snap_name, expected_attr in EXPECTED_SNAP_VARS.items():
            if snap_name in assignments and assignments[snap_name] != expected_attr:
                wrong[snap_name] = (expected_attr, assignments[snap_name])
        assert not wrong, f"Snapshot variables assigned from wrong globals attribute: {wrong}"

    def test_no_raw_globals_reads_in_loop_body(self, while_node):
        """
        After snapshotting, the loop body must not read the snapshotted globals
        directly.  The only allowed globals reads are:
        - target_path (written as a side-effect, not read)
        Other direct reads of snapshotted attributes should be replaced by snap_X.
        """
        reads_after_snapshot = _collect_globals_reads_after_snapshot(while_node)
        snapshotted_attrs = set(EXPECTED_SNAP_VARS.values())
        # target_path is a *write*, not a read — excluded by the Store-context filter.
        # Any remaining read of a snapshotted attribute is a violation.
        violations = snapshotted_attrs & reads_after_snapshot
        assert not violations, (
            f"Direct modules.globals reads found in while loop body for snapshotted "
            f"attributes (should use snap_X instead): {sorted(violations)}"
        )

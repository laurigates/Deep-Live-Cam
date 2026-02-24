"""Generate CLI argument documentation from modules/core.py using AST parsing.

Extracts all argparse add_argument() calls from parse_args(), skips deprecated
args (help=argparse.SUPPRESS), and outputs a markdown table. Can update README.md
in-place between marker comments or print to stdout.

Usage:
    uv run scripts/generate_cli_docs.py          # Update README.md in-place
    uv run scripts/generate_cli_docs.py --check   # Exit non-zero if README is stale
    uv run scripts/generate_cli_docs.py --stdout   # Print table to stdout
"""

import ast
import sys
from pathlib import Path

# Safe import — metadata.py contains only string literals
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.metadata import name as app_name, version as app_version

CORE_PY = Path(__file__).resolve().parent.parent / "modules" / "core.py"
README_MD = Path(__file__).resolve().parent.parent / "README.md"
START_MARKER = "<!-- CLI_ARGS_START -->"
END_MARKER = "<!-- CLI_ARGS_END -->"


def _resolve_constant(node: ast.expr) -> str:
    """Resolve an AST node to a display string for defaults/choices."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.List):
        items = [_resolve_constant(el) for el in node.elts]
        return ", ".join(items)
    # range(52) → "0-51", range(2, 11) → "2-10"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = node.args
        if len(args) == 1:
            stop = _resolve_constant(args[0])
            return f"0-{int(stop) - 1}"
        if len(args) >= 2:
            start = _resolve_constant(args[0])
            stop = _resolve_constant(args[1])
            return f"{start}-{int(stop) - 1}"
    # Function calls like suggest_max_memory() → "(auto)"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return "(auto)"
    if isinstance(node, ast.Attribute):
        # argparse.SUPPRESS
        if isinstance(node.value, ast.Name) and node.attr == "SUPPRESS":
            return "SUPPRESS"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_resolve_constant(node.operand)}"
    return str(ast.dump(node))


def extract_args(source: str) -> list[dict]:
    """Parse source code and extract add_argument() calls from parse_args()."""
    tree = ast.parse(source)

    # Find parse_args function
    parse_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            parse_fn = node
            break

    if parse_fn is None:
        raise ValueError("parse_args() function not found in source")

    args_list = []
    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue

        # Extract keyword arguments
        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = kw.value

        # Skip deprecated args (help=argparse.SUPPRESS)
        if "help" in kwargs:
            help_val = _resolve_constant(kwargs["help"])
            if help_val == "SUPPRESS":
                continue

        # Extract positional flag strings
        flags = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.append(arg.value)

        if not flags:
            continue

        entry = {"flags": ", ".join(flags)}

        # Help text
        if "help" in kwargs and isinstance(kwargs["help"], ast.Constant):
            entry["help"] = kwargs["help"].value
        else:
            entry["help"] = ""

        # Default
        if "default" in kwargs:
            entry["default"] = _resolve_constant(kwargs["default"])
        else:
            entry["default"] = ""

        # Choices
        if "choices" in kwargs:
            entry["choices"] = _resolve_constant(kwargs["choices"])
        else:
            entry["choices"] = ""

        # Action (store_true, version, etc.)
        if "action" in kwargs and isinstance(kwargs["action"], ast.Constant):
            entry["action"] = kwargs["action"].value
        else:
            entry["action"] = ""

        args_list.append(entry)

    return args_list


def format_table(args_list: list[dict]) -> str:
    """Format extracted args as a markdown table."""
    lines = []
    lines.append(f"*{app_name} v{app_version}* — run `uv run run.py --help` for latest options.")
    lines.append("")
    lines.append("| Flag | Description | Default | Choices |")
    lines.append("|------|-------------|---------|---------|")

    for arg in args_list:
        # Skip version action — not a real argument
        if arg["action"] == "version":
            continue

        flags = f"`{arg['flags']}`"
        desc = arg["help"]

        # Format default display
        default = arg["default"]
        if arg["action"] == "store_true":
            # Show actual default (usually False, but --keep-audio defaults True)
            if default == "True":
                default = "`True`"
            else:
                default = "`False`"
        elif default:
            default = f"`{default}`"

        choices = arg["choices"]
        if choices:
            choices = f"`{choices}`"

        lines.append(f"| {flags} | {desc} | {default} | {choices} |")

    return "\n".join(lines)


def generate_cli_docs() -> str:
    """Read core.py source and return the formatted CLI docs markdown."""
    source = CORE_PY.read_text()
    args_list = extract_args(source)
    return format_table(args_list)


def update_readme(table: str) -> bool:
    """Replace content between markers in README.md. Returns True if changed."""
    content = README_MD.read_text()

    if START_MARKER not in content or END_MARKER not in content:
        print(f"ERROR: Markers {START_MARKER} / {END_MARKER} not found in README.md", file=sys.stderr)
        return False

    before = content[: content.index(START_MARKER) + len(START_MARKER)]
    after = content[content.index(END_MARKER) :]

    new_content = f"{before}\n{table}\n{after}"

    if new_content == content:
        return False

    README_MD.write_text(new_content)
    return True


def main() -> None:
    table = generate_cli_docs()

    if "--stdout" in sys.argv:
        print(table)
        return

    if "--check" in sys.argv:
        content = README_MD.read_text()
        if START_MARKER not in content or END_MARKER not in content:
            print("ERROR: CLI_ARGS markers not found in README.md", file=sys.stderr)
            sys.exit(1)

        before = content[: content.index(START_MARKER) + len(START_MARKER)]
        after = content[content.index(END_MARKER) :]
        expected = f"{before}\n{table}\n{after}"

        if content != expected:
            print("README.md CLI args section is stale. Run: uv run scripts/generate_cli_docs.py", file=sys.stderr)
            sys.exit(1)

        print("README.md CLI args section is up to date.")
        return

    changed = update_readme(table)
    if changed:
        print("README.md updated with current CLI args.")
    else:
        print("README.md already up to date.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark baseline management tool.

Usage:
    # Save current benchmark results as baseline
    python scripts/benchmark_baseline.py save

    # Compare latest run against saved baseline
    python scripts/benchmark_baseline.py compare

    # Show saved baselines
    python scripts/benchmark_baseline.py list

    # Delete a specific baseline
    python scripts/benchmark_baseline.py delete <name>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def cmd_save(args):
    """Run benchmarks and save results as baseline."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running benchmarks...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/benchmarks/",
            "-m",
            "benchmark",
            "-v",
            "--tb=short",
            f"--benchmark-json={RESULTS_DIR / 'latest.json'}",
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\nBenchmarks exited with code {result.returncode}")
        if not args.force:
            print("Use --force to save anyway.")
            return 1

    results_path = RESULTS_DIR / "latest.json"
    if results_path.exists():
        data = json.loads(results_path.read_text())
        baseline_path = BASELINES_DIR / f"{args.name}.json"
        baseline_path.write_text(json.dumps(data, indent=2))
        print(f"\nBaseline saved: {baseline_path}")
    else:
        # Save a marker indicating benchmarks were run
        print("\nNote: pytest-benchmark JSON output not generated.")
        print("Baselines from conftest.BaselineManager can be saved within tests.")

    return 0


def cmd_compare(args):
    """Compare latest run against baseline."""
    baseline_path = BASELINES_DIR / f"{args.name}.json"
    if not baseline_path.exists():
        print(f"No baseline found: {baseline_path}")
        print(f"Run 'python {__file__} save --name {args.name}' first.")
        return 1

    baseline = json.loads(baseline_path.read_text())
    print(f"Baseline: {args.name}")
    print(f"  Saved: {baseline.get('timestamp', 'unknown')}")
    print(f"  Providers: {baseline.get('providers', 'unknown')}")

    if "results" in baseline:
        for key, val in baseline["results"].items():
            if isinstance(val, dict) and "mean_ms" in val:
                print(f"  {key}: {val['mean_ms']:.2f} ms ({val.get('fps', 0):.1f} FPS)")
    elif "benchmarks" in baseline:
        for bench in baseline["benchmarks"]:
            name = bench.get("name", "unknown")
            stats = bench.get("stats", {})
            mean = stats.get("mean", 0) * 1000
            print(f"  {name}: {mean:.2f} ms")

    print("\nRun benchmarks with: uv run pytest tests/benchmarks/ -m benchmark -v")
    print("Then compare results manually or use BaselineManager within tests.")
    return 0


def cmd_list(args):
    """List saved baselines."""
    if not BASELINES_DIR.exists():
        print("No baselines directory found.")
        return 0

    baselines = sorted(BASELINES_DIR.glob("*.json"))
    if not baselines:
        print("No baselines saved yet.")
        print(f"Run 'python {__file__} save' to create one.")
        return 0

    print("Saved baselines:")
    for path in baselines:
        data = json.loads(path.read_text())
        ts = data.get("timestamp", "unknown")
        providers = data.get("providers", [])
        print(f"  {path.stem}: saved {ts}, providers={providers}")
    return 0


def cmd_delete(args):
    """Delete a baseline."""
    path = BASELINES_DIR / f"{args.name}.json"
    if path.exists():
        path.unlink()
        print(f"Deleted: {path}")
    else:
        print(f"Not found: {path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline manager")
    sub = parser.add_subparsers(dest="command")

    save_p = sub.add_parser("save", help="Run benchmarks and save as baseline")
    save_p.add_argument("--name", default="default", help="Baseline name (default: 'default')")
    save_p.add_argument("--force", action="store_true", help="Save even if benchmarks fail")

    compare_p = sub.add_parser("compare", help="Compare against saved baseline")
    compare_p.add_argument("--name", default="default", help="Baseline name to compare against")

    sub.add_parser("list", help="List saved baselines")

    delete_p = sub.add_parser("delete", help="Delete a baseline")
    delete_p.add_argument("name", help="Baseline name to delete")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    handler = {
        "save": cmd_save,
        "compare": cmd_compare,
        "list": cmd_list,
        "delete": cmd_delete,
    }[args.command]

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

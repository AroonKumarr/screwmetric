#!/usr/bin/env python3
"""
ScrewMetric — Test Runner
==========================
Automatically discovers and runs all dataset processing tests,
then prints a human-readable summary showing total / passed / failed
counts and wall-clock execution time.

Usage::

    python run_tests.py               # run all tests
    python run_tests.py -v            # verbose (show each test name)
    python run_tests.py -k validator  # filter by keyword
    python run_tests.py --no-header   # suppress the banner

Authors: ScrewMetric Team
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade on Windows without ANSICON)
# ---------------------------------------------------------------------------

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _c(colour: str, text: str) -> str:
    """Wrap text in ANSI colour codes.

    Args:
        colour: One of the module-level colour constants.
        text: Text to colourise.

    Returns:
        Coloured string.
    """
    return f"{colour}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

@dataclass
class TestSummary:
    """Parsed summary from a pytest run.

    Attributes:
        total: Total number of tests collected.
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        errors: Number of tests that raised unexpected errors.
        skipped: Number of skipped tests.
        warnings: Number of warnings emitted.
        elapsed_seconds: Wall-clock time of the entire test run.
        exit_code: Process exit code returned by pytest.
        raw_output: Full raw stdout/stderr from pytest.
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    warnings: int = 0
    elapsed_seconds: float = 0.0
    exit_code: int = 0
    raw_output: str = ""

    @property
    def all_passed(self) -> bool:
        """True when no failures or errors were recorded."""
        return self.failed == 0 and self.errors == 0


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    import re
    return re.sub(r"\x1b\[[0-9;]*[mK]", "", text)


def _parse_pytest_output(output: str) -> dict[str, int]:
    """Extract numeric counts from pytest's final summary line.

    Pytest emits a line like:
    ``45 passed, 2 failed, 1 warning in 3.14s``

    Args:
        output: Full stdout+stderr from a pytest invocation.

    Returns:
        Dictionary mapping result types to counts.
    """
    counts: dict[str, int] = {
        "passed": 0, "failed": 0, "error": 0, "skipped": 0, "warning": 0,
    }
    for line in output.splitlines():
        stripped = _strip_ansi(line).strip()
        # Look for the final summary line: contains "passed" and is inside "=== ... ==="
        if not ("passed" in stripped or "failed" in stripped or "error" in stripped):
            continue
        # Strip surrounding "=" decorations
        clean = stripped.strip("=").strip()
        # Parse tokens like "123 passed", "2 failed", "1 warning"
        parts = [p.strip().rstrip(",") for p in clean.split()]
        i = 0
        while i < len(parts) - 1:
            token = parts[i]
            label = parts[i + 1].lower().rstrip("s")  # plurals
            if token.isdigit():
                count = int(token)
                for key in counts:
                    if label.startswith(key):
                        counts[key] = max(counts[key], count)
            i += 1
    return counts


# ---------------------------------------------------------------------------
# Banner / display helpers
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    width = 64
    print(f"\n{_BOLD}{_CYAN}╔{'═' * width}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╚{'═' * width}╝{_RESET}\n")


def _print_test_files(test_files: list[Path]) -> None:
    print(f"{_BOLD}Test modules discovered:{_RESET}")
    for f in sorted(test_files):
        print(f"  {_DIM}•{_RESET} {f.name}")
    print()


def _print_summary(summary: TestSummary, show_output: bool = False) -> None:
    """Render the final summary table to stdout.

    Args:
        summary: Parsed test result summary.
        show_output: If True, also print the raw pytest output.
    """
    status_icon = _c(_GREEN, "✅ ALL PASSED") if summary.all_passed else _c(_RED, "❌ FAILURES DETECTED")

    print(f"\n{_BOLD}{'─' * 66}{_RESET}")
    print(f"  {_BOLD}Test Run Summary{_RESET}")
    print(f"{'─' * 66}")
    print(f"  {'Status':<22}: {status_icon}")
    print(f"  {'Total tests':<22}: {_BOLD}{summary.total}{_RESET}")
    print(f"  {'Passed':<22}: {_c(_GREEN, str(summary.passed))}")

    if summary.failed:
        print(f"  {'Failed':<22}: {_c(_RED, str(summary.failed))}")
    else:
        print(f"  {'Failed':<22}: {summary.failed}")

    if summary.errors:
        print(f"  {'Errors':<22}: {_c(_RED, str(summary.errors))}")
    else:
        print(f"  {'Errors':<22}: {summary.errors}")

    if summary.skipped:
        print(f"  {'Skipped':<22}: {_c(_YELLOW, str(summary.skipped))}")
    else:
        print(f"  {'Skipped':<22}: {summary.skipped}")

    if summary.warnings:
        print(f"  {'Warnings':<22}: {_c(_YELLOW, str(summary.warnings))}")

    print(f"  {'Execution time':<22}: {summary.elapsed_seconds:.2f}s")
    print(f"  {'Exit code':<22}: {summary.exit_code}")
    print(f"{_BOLD}{'─' * 66}{_RESET}\n")

    if show_output and summary.raw_output:
        print(f"\n{_BOLD}Raw pytest output:{_RESET}")
        print(summary.raw_output)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def discover_test_files(tests_dir: Path) -> list[Path]:
    """Return all ``test_*.py`` files in ``tests_dir``.

    Args:
        tests_dir: Directory to search.

    Returns:
        Sorted list of test file paths.
    """
    return sorted(tests_dir.glob("test_*.py"))


def run_tests(
    tests_dir: Path,
    *,
    verbose: bool = False,
    keyword_filter: str = "",
    extra_args: list[str] | None = None,
) -> TestSummary:
    """Invoke pytest and return a parsed :class:`TestSummary`.

    Runs pytest once, streaming output to the terminal in real-time while
    also capturing it for summary parsing.

    Args:
        tests_dir: Directory containing test files.
        verbose: If True, pass ``-v`` to pytest for per-test output.
        keyword_filter: Passed as ``-k <filter>`` to pytest.
        extra_args: Any additional arguments forwarded to pytest.

    Returns:
        Populated :class:`TestSummary`.
    """
    import subprocess

    test_files = discover_test_files(tests_dir)
    if not test_files:
        print(_c(_YELLOW, f"⚠ No test files found in {tests_dir}"))
        return TestSummary()

    _print_test_files(test_files)

    cmd = [
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "--tb=short",
        "--no-header",
        "--color=yes",
    ]

    if verbose:
        cmd.append("-v")
    if keyword_filter:
        cmd.extend(["-k", keyword_filter])
    if extra_args:
        cmd.extend(extra_args)

    captured_lines: list[str] = []
    t_start = time.perf_counter()

    # Stream output live while also capturing each line
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(tests_dir.parent),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)
    proc.wait()

    elapsed = round(time.perf_counter() - t_start, 2)
    raw_output = "".join(captured_lines)
    counts = _parse_pytest_output(raw_output)

    total = sum(v for k, v in counts.items() if k != "warning")
    # If total is still 0, extract from "collected N items"
    if total == 0:
        for line in captured_lines:
            if "collected" in line and "item" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "collected" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        total = int(parts[i + 1])

    return TestSummary(
        total=total,
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["error"],
        skipped=counts["skipped"],
        warnings=counts["warning"],
        elapsed_seconds=elapsed,
        exit_code=proc.returncode,
        raw_output=raw_output,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_tests",
        description="ScrewMetric — Automated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_tests.py                    # run all tests\n"
            "  python run_tests.py -v                 # verbose\n"
            "  python run_tests.py -k validator       # filter by keyword\n"
            "  python run_tests.py --show-output      # show full pytest output\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Pass -v to pytest (show individual test names).",
    )
    parser.add_argument(
        "-k", "--keyword",
        default="",
        metavar="EXPR",
        help="Only run tests matching this expression (passed to pytest -k).",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print the full captured pytest output after the summary.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress the ScrewMetric banner.",
    )
    return parser


def main() -> None:
    """CLI entry point — parse args, run tests, print summary, set exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.no_header:
        _print_banner("ScrewMetric — Dataset Processing Test Suite")

    # Locate the tests directory relative to this script
    project_root = Path(__file__).resolve().parent
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print(_c(_RED, f"❌ Tests directory not found: {tests_dir}"))
        sys.exit(1)

    print(f"  Project root : {project_root}")
    print(f"  Tests dir    : {tests_dir}\n")

    summary = run_tests(
        tests_dir,
        verbose=args.verbose,
        keyword_filter=args.keyword,
    )

    _print_summary(summary, show_output=args.show_output)

    # Exit with pytest's exit code so CI systems can detect failures
    sys.exit(summary.exit_code)


if __name__ == "__main__":
    main()

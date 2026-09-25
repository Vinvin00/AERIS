"""Run unittest discovery and fail if an expected test file contains no tests."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests"
    if not target.exists():
        print(f"Required test target missing: {target}", file=sys.stderr)
        return 1
    suite = unittest.defaultTestLoader.discover(
        str(target if target.is_dir() else target.parent),
        pattern="test*.py" if target.is_dir() else target.name,
    )
    if suite.countTestCases() == 0:
        print(f"No tests discovered: {target}", file=sys.stderr)
        return 1
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

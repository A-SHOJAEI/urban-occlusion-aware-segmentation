#!/usr/bin/env python3
"""Check that all modules can be imported (syntax check)."""

import ast
import sys
from pathlib import Path


def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def main():
    """Main function to check all Python files."""
    root = Path(__file__).parent
    src_dir = root / "src"
    scripts_dir = root / "scripts"
    tests_dir = root / "tests"

    all_files = []
    all_files.extend(src_dir.rglob("*.py"))
    all_files.extend(scripts_dir.rglob("*.py"))
    all_files.extend(tests_dir.rglob("*.py"))

    print("=" * 70)
    print("SYNTAX CHECK")
    print("=" * 70)

    errors = []
    for file_path in sorted(all_files):
        rel_path = file_path.relative_to(root)
        valid, error = check_syntax(file_path)

        if valid:
            print(f"✓ {rel_path}")
        else:
            print(f"✗ {rel_path}")
            print(f"  Error: {error}")
            errors.append((rel_path, error))

    print("\n" + "=" * 70)
    if errors:
        print(f"✗ Found {len(errors)} syntax errors:")
        for rel_path, error in errors:
            print(f"  - {rel_path}: {error}")
        return 1
    else:
        print(f"✓ All {len(all_files)} Python files have valid syntax!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

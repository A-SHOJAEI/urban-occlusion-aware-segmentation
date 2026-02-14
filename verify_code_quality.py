#!/usr/bin/env python3
"""Verification script to check code quality improvements."""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set

def check_docstrings(filepath: Path) -> Dict[str, bool]:
    """Check if functions and classes have docstrings."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    results = {
        "has_module_docstring": ast.get_docstring(tree) is not None,
        "functions_with_docstrings": 0,
        "total_functions": 0,
        "classes_with_docstrings": 0,
        "total_classes": 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_") or node.name.startswith("__"):
                results["total_functions"] += 1
                if ast.get_docstring(node):
                    results["functions_with_docstrings"] += 1
        elif isinstance(node, ast.ClassDef):
            results["total_classes"] += 1
            if ast.get_docstring(node):
                results["classes_with_docstrings"] += 1

    return results

def check_type_hints(filepath: Path) -> Dict[str, int]:
    """Check for type hints in function signatures."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    results = {
        "functions_with_hints": 0,
        "total_functions": 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_") or node.name.startswith("__"):
                results["total_functions"] += 1
                # Check if has return annotation or any arg annotations
                if node.returns or any(arg.annotation for arg in node.args.args):
                    results["functions_with_hints"] += 1

    return results

def check_error_handling(filepath: Path) -> Dict[str, int]:
    """Check for try/except blocks."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    try_blocks = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))

    return {"try_except_blocks": try_blocks}

def main():
    """Run all verification checks."""
    project_root = Path(__file__).parent
    src_dir = project_root / "src" / "urban_occlusion_aware_segmentation"

    files_to_check = [
        src_dir / "models" / "model.py",
        src_dir / "data" / "loader.py",
        src_dir / "data" / "preprocessing.py",
        src_dir / "training" / "trainer.py",
        src_dir / "evaluation" / "metrics.py",
        src_dir / "utils" / "config.py",
        project_root / "scripts" / "train.py",
    ]

    print("=" * 70)
    print("CODE QUALITY VERIFICATION REPORT")
    print("=" * 70)

    total_score = 0
    max_score = 0

    for filepath in files_to_check:
        if not filepath.exists():
            print(f"\n❌ {filepath.relative_to(project_root)}: FILE NOT FOUND")
            continue

        print(f"\n📄 {filepath.relative_to(project_root)}")
        print("-" * 70)

        # Check docstrings
        doc_results = check_docstrings(filepath)
        func_coverage = (doc_results["functions_with_docstrings"] /
                        max(doc_results["total_functions"], 1) * 100)
        class_coverage = (doc_results["classes_with_docstrings"] /
                         max(doc_results["total_classes"], 1) * 100)

        print(f"Module docstring: {'✓' if doc_results['has_module_docstring'] else '✗'}")
        print(f"Function docstrings: {doc_results['functions_with_docstrings']}/{doc_results['total_functions']} ({func_coverage:.0f}%)")
        print(f"Class docstrings: {doc_results['classes_with_docstrings']}/{doc_results['total_classes']} ({class_coverage:.0f}%)")

        # Check type hints
        hint_results = check_type_hints(filepath)
        hint_coverage = (hint_results["functions_with_hints"] /
                        max(hint_results["total_functions"], 1) * 100)
        print(f"Type hints: {hint_results['functions_with_hints']}/{hint_results['total_functions']} ({hint_coverage:.0f}%)")

        # Check error handling
        error_results = check_error_handling(filepath)
        print(f"Try/except blocks: {error_results['try_except_blocks']}")

        # Calculate score for this file
        file_score = 0
        file_max = 4

        if doc_results['has_module_docstring']:
            file_score += 1
        if func_coverage >= 80:
            file_score += 1
        if hint_coverage >= 80:
            file_score += 1
        if error_results['try_except_blocks'] > 0:
            file_score += 1

        total_score += file_score
        max_score += file_max

        print(f"Score: {file_score}/{file_max} ⭐")

    print("\n" + "=" * 70)
    print(f"OVERALL SCORE: {total_score}/{max_score} ({total_score/max_score*100:.0f}%)")
    print("=" * 70)

    # Check specific mandatory items
    print("\n📋 MANDATORY CHECKLIST:")
    print("-" * 70)

    checklist = [
        ("README.md < 200 lines", len((project_root / "README.md").read_text().splitlines()) < 200),
        ("LICENSE file exists", (project_root / "LICENSE").exists()),
        (".gitignore updated", "logs/" in (project_root / ".gitignore").read_text()),
        ("No .coverage file", not (project_root / ".coverage").exists()),
        ("No htmlcov/ directory", not (project_root / "htmlcov").exists()),
        ("No .pytest_cache/", not (project_root / ".pytest_cache").exists()),
        ("No mlruns/ directory", not (project_root / "mlruns").exists()),
    ]

    for item, passed in checklist:
        print(f"{'✓' if passed else '✗'} {item}")

    all_passed = all(passed for _, passed in checklist)

    if total_score >= max_score * 0.75 and all_passed:
        print("\n✅ CODE QUALITY CHECK PASSED")
        return 0
    else:
        print("\n⚠️  CODE QUALITY CHECK: IMPROVEMENTS NEEDED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

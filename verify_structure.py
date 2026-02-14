#!/usr/bin/env python3
"""Verify project structure is complete."""

import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING {description}: {path}")
        return False

def check_dir(path, description):
    """Check if a directory exists."""
    if path.exists() and path.is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING {description}: {path}")
        return False

def main():
    """Main verification function."""
    root = Path(__file__).parent
    all_checks = []

    print("=" * 70)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 70)

    # Core files
    print("\n📄 Core Files:")
    all_checks.append(check_file(root / "README.md", "README"))
    all_checks.append(check_file(root / "LICENSE", "LICENSE"))
    all_checks.append(check_file(root / "requirements.txt", "Requirements"))
    all_checks.append(check_file(root / "pyproject.toml", "PyProject"))
    all_checks.append(check_file(root / ".gitignore", "GitIgnore"))

    # Configuration
    print("\n⚙️  Configuration:")
    all_checks.append(check_file(root / "configs" / "default.yaml", "Default Config"))

    # Source code
    print("\n📦 Source Code:")
    src = root / "src" / "urban_occlusion_aware_segmentation"
    all_checks.append(check_file(src / "__init__.py", "Package Init"))
    all_checks.append(check_file(src / "data" / "__init__.py", "Data Module Init"))
    all_checks.append(check_file(src / "data" / "loader.py", "Data Loader"))
    all_checks.append(check_file(src / "data" / "preprocessing.py", "Data Preprocessing"))
    all_checks.append(check_file(src / "models" / "__init__.py", "Models Module Init"))
    all_checks.append(check_file(src / "models" / "model.py", "Model Implementation"))
    all_checks.append(check_file(src / "training" / "__init__.py", "Training Module Init"))
    all_checks.append(check_file(src / "training" / "trainer.py", "Trainer"))
    all_checks.append(check_file(src / "evaluation" / "__init__.py", "Evaluation Module Init"))
    all_checks.append(check_file(src / "evaluation" / "metrics.py", "Metrics"))
    all_checks.append(check_file(src / "utils" / "__init__.py", "Utils Module Init"))
    all_checks.append(check_file(src / "utils" / "config.py", "Config Utils"))

    # Scripts
    print("\n🚀 Scripts:")
    all_checks.append(check_file(root / "scripts" / "train.py", "Training Script"))
    all_checks.append(check_file(root / "scripts" / "evaluate.py", "Evaluation Script"))

    # Tests
    print("\n🧪 Tests:")
    all_checks.append(check_file(root / "tests" / "__init__.py", "Tests Init"))
    all_checks.append(check_file(root / "tests" / "conftest.py", "Test Fixtures"))
    all_checks.append(check_file(root / "tests" / "test_data.py", "Data Tests"))
    all_checks.append(check_file(root / "tests" / "test_model.py", "Model Tests"))
    all_checks.append(check_file(root / "tests" / "test_training.py", "Training Tests"))

    # Notebooks
    print("\n📓 Notebooks:")
    all_checks.append(check_file(root / "notebooks" / "exploration.ipynb", "Exploration Notebook"))

    # Directories
    print("\n📁 Directories:")
    all_checks.append(check_dir(root / "models", "Models Directory"))
    all_checks.append(check_dir(root / "checkpoints", "Checkpoints Directory"))
    all_checks.append(check_dir(root / "results", "Results Directory"))
    all_checks.append(check_dir(root / "logs", "Logs Directory"))

    # Summary
    print("\n" + "=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100

    print(f"SUMMARY: {passed}/{total} checks passed ({percentage:.1f}%)")

    if passed == total:
        print("✓ All files and directories present!")
        return 0
    else:
        print(f"✗ Missing {total - passed} required files/directories")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Basic import test to verify module structure."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing basic imports...")

try:
    print("1. Testing utils imports...")
    from urban_occlusion_aware_segmentation.utils.config import load_config, get_device
    print("   ✓ Utils imports successful")
except Exception as e:
    print(f"   ✗ Utils imports failed: {e}")
    sys.exit(1)

try:
    print("2. Testing data imports...")
    from urban_occlusion_aware_segmentation.data.preprocessing import compute_boundary_mask
    print("   ✓ Data preprocessing imports successful")
except Exception as e:
    print(f"   ✗ Data preprocessing imports failed: {e}")
    sys.exit(1)

try:
    print("3. Testing model imports (may warn about missing dependencies)...")
    from urban_occlusion_aware_segmentation.models.model import DeepLabV3PlusSegmentation
    print("   ✓ Model imports successful")
except Exception as e:
    print(f"   ✗ Model imports failed: {e}")
    sys.exit(1)

try:
    print("4. Testing training imports...")
    from urban_occlusion_aware_segmentation.training.trainer import OcclusionWeightedLoss
    print("   ✓ Training imports successful")
except Exception as e:
    print(f"   ✗ Training imports failed: {e}")
    sys.exit(1)

print("\n✅ All basic imports successful!")

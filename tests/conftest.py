"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root and src/ to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Provide torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def num_classes() -> int:
    """Number of segmentation classes."""
    return 19


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Generate a sample RGB image tensor."""
    return torch.randn(2, 3, 256, 512)  # Batch of 2


@pytest.fixture
def sample_mask(num_classes: int) -> torch.Tensor:
    """Generate a sample segmentation mask."""
    return torch.randint(0, num_classes, (2, 256, 512))


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration."""
    return {
        "data": {
            "num_classes": 19,
            "image_size": [256, 512],
            "use_synthetic": True,
        },
        "model": {
            "type": "ensemble",
            "segformer": {
                "backbone": "mit_b3",
                "pretrained": False,
                "dropout": 0.1,
            },
            "deeplabv3plus": {
                "backbone": "resnet101",
                "pretrained": False,
                "output_stride": 16,
            },
            "ensemble": {
                "weights": [0.5, 0.5],
                "uncertainty_threshold": 0.3,
            },
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "gradient_clip": 1.0,
            "amp": False,
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001,
            },
            "save_dir": "./test_checkpoints",
            "save_freq": 1,
        },
        "loss": {
            "primary": "occlusion_weighted_ce",
            "auxiliary": "boundary_loss",
            "weights": {
                "primary": 0.7,
                "auxiliary": 0.3,
            },
            "occlusion": {
                "boundary_width": 5,
                "boundary_weight": 3.0,
                "edge_threshold": 0.1,
            },
        },
        "augmentation": {
            "train": {
                "horizontal_flip": 0.5,
                "random_scale": [0.5, 2.0],
                "random_crop": True,
                "color_jitter": {
                    "brightness": 0.3,
                    "contrast": 0.3,
                    "saturation": 0.3,
                    "hue": 0.1,
                },
                "random_rotate": 10,
                "gaussian_blur": 0.2,
            },
            "val": {
                "resize": [256, 512],
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "evaluation": {
            "batch_size": 2,
            "metrics": ["miou", "boundary_f1", "occlusion_recall", "pixel_accuracy"],
            "boundary_threshold": 5,
            "occlusion_classes": [11, 12, 13, 14],
        },
        "system": {
            "seed": 42,
            "num_workers": 0,
            "pin_memory": False,
            "device": "cpu",
        },
        "logging": {
            "log_dir": "./test_logs",
            "log_interval": 1,
            "tensorboard": False,
            "mlflow": {
                "tracking_uri": "./test_mlruns",
                "experiment_name": "test_experiment",
            },
        },
    }


@pytest.fixture
def sample_predictions(num_classes: int) -> np.ndarray:
    """Generate sample predictions for metric testing."""
    np.random.seed(42)
    return np.random.randint(0, num_classes, (2, 256, 512))


@pytest.fixture
def sample_targets(num_classes: int) -> np.ndarray:
    """Generate sample ground truth masks for metric testing."""
    np.random.seed(42)
    return np.random.randint(0, num_classes, (2, 256, 512))

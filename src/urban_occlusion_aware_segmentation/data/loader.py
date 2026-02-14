"""Data loading utilities for urban segmentation datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes

from urban_occlusion_aware_segmentation.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger(__name__)


class CityscapesDataset(Dataset):
    """Wrapper for Cityscapes dataset with custom transformations.

    Args:
        root: Root directory of the dataset.
        split: Dataset split ('train', 'val', or 'test').
        transforms: Albumentations transforms to apply.
        target_type: Type of target ('semantic' or 'instance').
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Any] = None,
        target_type: str = "semantic",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.target_type = target_type

        # Try to use torchvision Cityscapes if available
        try:
            self.dataset = Cityscapes(
                root=str(self.root),
                split=split,
                mode="fine",
                target_type=target_type,
            )
            self.use_cityscapes = True
            logger.info(f"Loaded Cityscapes dataset: {split} split with {len(self.dataset)} samples")
        except (RuntimeError, FileNotFoundError):
            logger.warning(
                "Cityscapes dataset not found. Please download it or use synthetic data."
            )
            self.use_cityscapes = False
            self.dataset = []

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset) if self.use_cityscapes else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, segmentation_mask).
        """
        if not self.use_cityscapes:
            raise RuntimeError("Dataset not available")

        image, target = self.dataset[idx]

        # Convert PIL images to numpy arrays
        image = np.array(image)
        target = np.array(target)

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=image, mask=target)
            image = transformed["image"]
            target = transformed["mask"]

        return image, target.long()


class SyntheticUrbanDataset(Dataset):
    """Synthetic dataset for testing and development.

    Generates random images and segmentation masks for rapid prototyping.

    Args:
        num_samples: Number of synthetic samples to generate.
        image_size: Tuple of (height, width) for images.
        num_classes: Number of segmentation classes.
        transforms: Albumentations transforms to apply.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 19,
        transforms: Optional[Any] = None,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transforms = transforms

        logger.info(
            f"Created synthetic dataset with {num_samples} samples, "
            f"size {image_size}, {num_classes} classes"
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic sample.

        Args:
            idx: Sample index (used as random seed).

        Returns:
            Tuple of (image, segmentation_mask).
        """
        # Set seed for reproducibility
        rng = np.random.RandomState(idx)

        # Generate random RGB image
        image = rng.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)

        # Generate segmentation mask with realistic structure
        # Create blocks of same class to simulate objects
        block_size = 64
        mask = np.zeros(self.image_size, dtype=np.uint8)

        for i in range(0, self.image_size[0], block_size):
            for j in range(0, self.image_size[1], block_size):
                class_id = rng.randint(0, self.num_classes)
                mask[
                    i : min(i + block_size, self.image_size[0]),
                    j : min(j + block_size, self.image_size[1]),
                ] = class_id

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Convert to tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        return image, mask.long()


def get_data_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with error handling.

    Args:
        config: Configuration dictionary containing data, training, and system settings.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If dataset creation or data loader initialization fails.
    """
    try:
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        system_config = config.get("system", {})

        root_dir = data_config.get("root_dir", "./data")
        num_classes = data_config.get("num_classes", 19)
        image_size = tuple(data_config.get("image_size", [512, 1024]))
        use_synthetic = data_config.get("use_synthetic", True)

        batch_size = training_config.get("batch_size", 4)
        num_workers = system_config.get("num_workers", 4)
        pin_memory = system_config.get("pin_memory", True)

        # Validate configuration
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be positive.")
        if num_classes <= 0:
            raise ValueError(f"Invalid num_classes: {num_classes}. Must be positive.")
        if len(image_size) != 2 or any(s <= 0 for s in image_size):
            raise ValueError(f"Invalid image_size: {image_size}. Must be (H, W) with positive values.")

        logger.info(
            f"Creating data loaders: batch_size={batch_size}, "
            f"image_size={image_size}, num_classes={num_classes}"
        )

        # Get transforms
        try:
            train_transforms = get_train_transforms(config)
            val_transforms = get_val_transforms(config)
        except Exception as e:
            logger.error(f"Failed to create transforms: {e}")
            raise RuntimeError(f"Transform creation failed: {e}") from e

        # Create datasets
        try:
            if use_synthetic:
                logger.info("Using synthetic data for training")
                train_dataset = SyntheticUrbanDataset(
                    num_samples=200,
                    image_size=image_size,
                    num_classes=num_classes,
                    transforms=train_transforms,
                )
                val_dataset = SyntheticUrbanDataset(
                    num_samples=50,
                    image_size=image_size,
                    num_classes=num_classes,
                    transforms=val_transforms,
                )
            else:
                logger.info("Using Cityscapes dataset")
                train_dataset = CityscapesDataset(
                    root=root_dir,
                    split="train",
                    transforms=train_transforms,
                )
                val_dataset = CityscapesDataset(
                    root=root_dir,
                    split="val",
                    transforms=val_transforms,
                )

                # Check if datasets are empty
                if len(train_dataset) == 0:
                    raise RuntimeError(
                        f"No training data found at {root_dir}. "
                        f"Please download Cityscapes or set use_synthetic=true."
                    )
                if len(val_dataset) == 0:
                    raise RuntimeError(
                        f"No validation data found at {root_dir}. "
                        f"Please download Cityscapes or set use_synthetic=true."
                    )

        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            raise RuntimeError(f"Dataset creation failed: {e}") from e

        # Create data loaders
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            logger.info(
                f"Successfully created data loaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches"
            )

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise RuntimeError(f"DataLoader creation failed: {e}") from e

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_data_loaders: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create data loaders: {e}") from e

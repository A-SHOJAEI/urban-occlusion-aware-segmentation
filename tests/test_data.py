"""Tests for data loading and preprocessing modules."""

import numpy as np
import pytest
import torch

from urban_occlusion_aware_segmentation.data.loader import (
    SyntheticUrbanDataset,
    get_data_loaders,
)
from urban_occlusion_aware_segmentation.data.preprocessing import (
    compute_boundary_mask,
    compute_edge_weights,
    get_train_transforms,
    get_val_transforms,
)


class TestSyntheticUrbanDataset:
    """Test suite for SyntheticUrbanDataset."""

    def test_dataset_length(self) -> None:
        """Test that dataset returns correct length."""
        num_samples = 50
        dataset = SyntheticUrbanDataset(num_samples=num_samples)
        assert len(dataset) == num_samples

    def test_dataset_getitem(self) -> None:
        """Test that dataset returns correct data types and shapes."""
        dataset = SyntheticUrbanDataset(
            num_samples=10, image_size=(256, 512), num_classes=19
        )

        image, mask = dataset[0]

        # Check types
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

        # Check shapes
        assert image.shape == (3, 256, 512)
        assert mask.shape == (256, 512)

        # Check value ranges
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert mask.min() >= 0 and mask.max() < 19

    def test_dataset_reproducibility(self) -> None:
        """Test that same index returns same sample."""
        dataset = SyntheticUrbanDataset(num_samples=10)

        image1, mask1 = dataset[5]
        image2, mask2 = dataset[5]

        assert torch.allclose(image1, image2)
        assert torch.equal(mask1, mask2)


class TestTransforms:
    """Test suite for data transformations."""

    def test_train_transforms(self, sample_config: dict) -> None:
        """Test that train transforms are created correctly."""
        transforms = get_train_transforms(sample_config)
        assert transforms is not None

    def test_val_transforms(self, sample_config: dict) -> None:
        """Test that validation transforms are created correctly."""
        transforms = get_val_transforms(sample_config)
        assert transforms is not None

    def test_train_transforms_output(self, sample_config: dict) -> None:
        """Test that transforms produce correct output shapes."""
        transforms = get_train_transforms(sample_config)

        # Create dummy data
        image = np.random.randint(0, 256, (512, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (512, 1024), dtype=np.uint8)

        # Apply transforms
        transformed = transforms(image=image, mask=mask)

        assert "image" in transformed
        assert "mask" in transformed

        # Check output types
        assert isinstance(transformed["image"], torch.Tensor)
        assert isinstance(transformed["mask"], torch.Tensor)


class TestPreprocessing:
    """Test suite for preprocessing utilities."""

    def test_compute_boundary_mask(self) -> None:
        """Test boundary mask computation."""
        # Create simple segmentation with clear boundary
        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[40:60, 40:60] = 1

        boundary = compute_boundary_mask(segmentation, width=5)

        # Check that boundaries are detected
        assert boundary.sum() > 0

        # Check that interior and far exterior have no boundary
        assert boundary[50, 50] == 0  # Interior
        assert boundary[0, 0] == 0  # Far exterior

    def test_compute_edge_weights(self) -> None:
        """Test edge weight computation."""
        segmentation = np.zeros((100, 100), dtype=np.uint8)
        segmentation[40:60, 40:60] = 1

        weights = compute_edge_weights(
            segmentation, boundary_width=5, boundary_weight=3.0
        )

        # Check that weights are reasonable
        assert weights.min() >= 1.0
        assert weights.max() <= 3.0
        assert weights.shape == segmentation.shape


class TestDataLoaders:
    """Test suite for data loaders."""

    def test_get_data_loaders(self, sample_config: dict) -> None:
        """Test that data loaders are created correctly."""
        train_loader, val_loader = get_data_loaders(sample_config)

        assert train_loader is not None
        assert val_loader is not None

        # Check batch sizes
        assert train_loader.batch_size == sample_config["training"]["batch_size"]

    def test_data_loader_iteration(self, sample_config: dict) -> None:
        """Test that data loaders can be iterated."""
        train_loader, _ = get_data_loaders(sample_config)

        # Get one batch
        batch = next(iter(train_loader))
        images, masks = batch

        # Check shapes
        batch_size = sample_config["training"]["batch_size"]
        assert images.shape[0] == batch_size
        assert masks.shape[0] == batch_size

        # Check types
        assert isinstance(images, torch.Tensor)
        assert isinstance(masks, torch.Tensor)

"""Tests for model implementations."""

import pytest
import torch

from urban_occlusion_aware_segmentation.models.model import (
    DeepLabV3PlusSegmentation,
    OcclusionAwareEnsemble,
    SegFormerSegmentation,
    build_model,
)


class TestSegFormerSegmentation:
    """Test suite for SegFormer model."""

    def test_model_initialization(self, num_classes: int) -> None:
        """Test that SegFormer initializes correctly."""
        model = SegFormerSegmentation(
            num_classes=num_classes, backbone="mit_b3", pretrained=False
        )
        assert model is not None
        assert model.num_classes == num_classes

    def test_forward_pass(self, sample_image: torch.Tensor, num_classes: int) -> None:
        """Test forward pass produces correct output shape."""
        model = SegFormerSegmentation(num_classes=num_classes, pretrained=False)
        model.eval()

        with torch.no_grad():
            output = model(sample_image)

        # Check output shape
        batch_size, _, height, width = sample_image.shape
        assert output.shape == (batch_size, num_classes, height, width)

    def test_model_training_mode(self, sample_image: torch.Tensor, num_classes: int) -> None:
        """Test that model can switch between train and eval modes."""
        model = SegFormerSegmentation(num_classes=num_classes, pretrained=False)

        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False


class TestDeepLabV3PlusSegmentation:
    """Test suite for DeepLabV3+ model."""

    def test_model_initialization(self, num_classes: int) -> None:
        """Test that DeepLabV3+ initializes correctly."""
        model = DeepLabV3PlusSegmentation(
            num_classes=num_classes, backbone="resnet101", pretrained=False
        )
        assert model is not None
        assert model.num_classes == num_classes

    def test_forward_pass(self, sample_image: torch.Tensor, num_classes: int) -> None:
        """Test forward pass produces correct output shape."""
        model = DeepLabV3PlusSegmentation(num_classes=num_classes, pretrained=False)
        model.eval()

        with torch.no_grad():
            output = model(sample_image)

        batch_size, _, height, width = sample_image.shape
        assert output.shape == (batch_size, num_classes, height, width)


class TestOcclusionAwareEnsemble:
    """Test suite for ensemble model."""

    def test_model_initialization(self, num_classes: int) -> None:
        """Test that ensemble initializes correctly."""
        model = OcclusionAwareEnsemble(
            num_classes=num_classes,
            segformer_config={"pretrained": False},
            deeplabv3_config={"pretrained": False},
        )
        assert model is not None
        assert model.num_classes == num_classes

    def test_forward_pass(self, sample_image: torch.Tensor, num_classes: int) -> None:
        """Test forward pass without uncertainty."""
        model = OcclusionAwareEnsemble(
            num_classes=num_classes,
            segformer_config={"pretrained": False},
            deeplabv3_config={"pretrained": False},
        )
        model.eval()

        with torch.no_grad():
            output = model(sample_image, return_uncertainty=False)

        batch_size, _, height, width = sample_image.shape
        assert output.shape == (batch_size, num_classes, height, width)

    def test_forward_pass_with_uncertainty(
        self, sample_image: torch.Tensor, num_classes: int
    ) -> None:
        """Test forward pass with uncertainty map."""
        model = OcclusionAwareEnsemble(
            num_classes=num_classes,
            segformer_config={"pretrained": False},
            deeplabv3_config={"pretrained": False},
        )
        model.eval()

        with torch.no_grad():
            output, uncertainty = model(sample_image, return_uncertainty=True)

        batch_size, _, height, width = sample_image.shape
        assert output.shape == (batch_size, num_classes, height, width)
        assert uncertainty.shape == (batch_size, height, width)

        # Check uncertainty is in valid range
        assert uncertainty.min() >= 0.0
        assert uncertainty.max() <= 1.0

    def test_uncertainty_computation(
        self, sample_image: torch.Tensor, num_classes: int
    ) -> None:
        """Test that uncertainty is computed correctly."""
        model = OcclusionAwareEnsemble(
            num_classes=num_classes,
            segformer_config={"pretrained": False},
            deeplabv3_config={"pretrained": False},
        )

        # Create two different predictions
        pred1 = torch.randn(2, num_classes, 256, 512)
        pred2 = torch.randn(2, num_classes, 256, 512)

        uncertainty = model._compute_uncertainty(pred1, pred2)

        # Check shape and range
        assert uncertainty.shape == (2, 256, 512)
        assert uncertainty.min() >= 0.0
        assert uncertainty.max() <= 1.0


class TestBuildModel:
    """Test suite for model building utility."""

    def test_build_segformer(self, sample_config: dict, device: torch.device) -> None:
        """Test building SegFormer from config."""
        config = sample_config.copy()
        config["model"]["type"] = "segformer"
        config["model"]["segformer"]["pretrained"] = False

        model = build_model(config, device)

        assert isinstance(model, SegFormerSegmentation)

    def test_build_deeplabv3plus(
        self, sample_config: dict, device: torch.device
    ) -> None:
        """Test building DeepLabV3+ from config."""
        config = sample_config.copy()
        config["model"]["type"] = "deeplabv3plus"
        config["model"]["deeplabv3plus"]["pretrained"] = False

        model = build_model(config, device)

        assert isinstance(model, DeepLabV3PlusSegmentation)

    def test_build_ensemble(self, sample_config: dict, device: torch.device) -> None:
        """Test building ensemble from config."""
        config = sample_config.copy()
        config["model"]["type"] = "ensemble"
        config["model"]["segformer"]["pretrained"] = False
        config["model"]["deeplabv3plus"]["pretrained"] = False

        model = build_model(config, device)

        assert isinstance(model, OcclusionAwareEnsemble)

    def test_build_invalid_model_type(
        self, sample_config: dict, device: torch.device
    ) -> None:
        """Test that invalid model type raises error."""
        config = sample_config.copy()
        config["model"]["type"] = "invalid_model"

        with pytest.raises(ValueError):
            build_model(config, device)


class TestModelGradients:
    """Test suite for gradient flow."""

    def test_segformer_gradients(
        self, sample_image: torch.Tensor, sample_mask: torch.Tensor, num_classes: int
    ) -> None:
        """Test that gradients flow through SegFormer."""
        model = SegFormerSegmentation(num_classes=num_classes, pretrained=False)
        model.train()

        output = model(sample_image)
        loss = torch.nn.functional.cross_entropy(output, sample_mask)
        loss.backward()

        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_ensemble_gradients(
        self, sample_image: torch.Tensor, sample_mask: torch.Tensor, num_classes: int
    ) -> None:
        """Test that gradients flow through ensemble."""
        model = OcclusionAwareEnsemble(
            num_classes=num_classes,
            segformer_config={"pretrained": False},
            deeplabv3_config={"pretrained": False},
        )
        model.train()

        output = model(sample_image)
        loss = torch.nn.functional.cross_entropy(output, sample_mask)
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

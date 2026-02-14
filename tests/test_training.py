"""Tests for training modules."""

import shutil
from pathlib import Path

import pytest
import torch

from urban_occlusion_aware_segmentation.data.loader import get_data_loaders
from urban_occlusion_aware_segmentation.models.model import build_model
from urban_occlusion_aware_segmentation.training.trainer import (
    BoundaryLoss,
    OcclusionWeightedLoss,
    Trainer,
)


class TestOcclusionWeightedLoss:
    """Test suite for custom loss function."""

    def test_loss_initialization(self, num_classes: int) -> None:
        """Test loss function initialization."""
        loss_fn = OcclusionWeightedLoss(num_classes=num_classes)
        assert loss_fn is not None
        assert loss_fn.num_classes == num_classes

    def test_loss_forward(
        self, sample_image: torch.Tensor, sample_mask: torch.Tensor, num_classes: int
    ) -> None:
        """Test loss computation."""
        loss_fn = OcclusionWeightedLoss(num_classes=num_classes)

        # Create fake predictions
        predictions = torch.randn(
            sample_mask.shape[0], num_classes, sample_mask.shape[1], sample_mask.shape[2]
        )

        loss = loss_fn(predictions, sample_mask)

        # Check that loss is scalar and positive
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_loss_backprop(
        self, sample_image: torch.Tensor, sample_mask: torch.Tensor, num_classes: int
    ) -> None:
        """Test that loss allows backpropagation."""
        loss_fn = OcclusionWeightedLoss(num_classes=num_classes)

        predictions = torch.randn(
            sample_mask.shape[0],
            num_classes,
            sample_mask.shape[1],
            sample_mask.shape[2],
            requires_grad=True,
        )

        loss = loss_fn(predictions, sample_mask)
        loss.backward()

        # Check that gradients exist
        assert predictions.grad is not None


class TestBoundaryLoss:
    """Test suite for boundary loss."""

    def test_loss_initialization(self, num_classes: int) -> None:
        """Test boundary loss initialization."""
        loss_fn = BoundaryLoss(num_classes=num_classes)
        assert loss_fn is not None

    def test_loss_forward(self, sample_mask: torch.Tensor, num_classes: int) -> None:
        """Test boundary loss computation."""
        loss_fn = BoundaryLoss(num_classes=num_classes)

        predictions = torch.randn(
            sample_mask.shape[0], num_classes, sample_mask.shape[1], sample_mask.shape[2]
        )

        loss = loss_fn(predictions, sample_mask)

        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestTrainer:
    """Test suite for Trainer class."""

    @pytest.fixture
    def trainer(self, sample_config: dict, device: torch.device) -> Trainer:
        """Create trainer instance for testing."""
        # Modify config for faster testing
        config = sample_config.copy()
        config["training"]["num_epochs"] = 2
        config["training"]["amp"] = False
        config["model"]["segformer"]["pretrained"] = False
        config["model"]["deeplabv3plus"]["pretrained"] = False

        # Build model
        model = build_model(config, device)

        # Get data loaders
        train_loader, val_loader = get_data_loaders(config)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        yield trainer

        # Cleanup
        save_dir = Path(config["training"]["save_dir"])
        if save_dir.exists():
            shutil.rmtree(save_dir)

    def test_trainer_initialization(self, trainer: Trainer) -> None:
        """Test that trainer initializes correctly."""
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_epoch(self, trainer: Trainer) -> None:
        """Test single training epoch."""
        loss = trainer.train_epoch()

        # Check that loss is reasonable
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_validate(self, trainer: Trainer) -> None:
        """Test validation."""
        val_loss = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss >= 0.0

    def test_save_checkpoint(self, trainer: Trainer) -> None:
        """Test checkpoint saving."""
        trainer.save_checkpoint(epoch=0, val_loss=1.0, is_best=True)

        checkpoint_path = trainer.save_dir / "best_model.pth"
        assert checkpoint_path.exists()

        # Check that checkpoint contains required keys
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint

    def test_load_checkpoint(self, trainer: Trainer) -> None:
        """Test checkpoint loading."""
        # Save checkpoint first
        trainer.save_checkpoint(epoch=0, val_loss=1.0, is_best=True)

        checkpoint_path = trainer.save_dir / "best_model.pth"

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        # Verify loading
        assert trainer.current_epoch == 0

    def test_full_training_loop(self, trainer: Trainer) -> None:
        """Test complete training loop."""
        # This is an integration test
        trainer.num_epochs = 2
        trainer.train()

        # Check that training completed
        assert len(trainer.train_losses) == 2
        assert len(trainer.val_losses) == 2

        # Check that best model was saved
        best_model_path = trainer.save_dir / "best_model.pth"
        assert best_model_path.exists()


class TestOptimizers:
    """Test suite for optimizer building."""

    def test_adamw_optimizer(self, sample_config: dict, device: torch.device) -> None:
        """Test AdamW optimizer creation."""
        config = sample_config.copy()
        config["training"]["optimizer"] = "adamw"
        config["model"]["segformer"]["pretrained"] = False
        config["model"]["deeplabv3plus"]["pretrained"] = False

        model = build_model(config, device)
        train_loader, val_loader = get_data_loaders(config)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        assert isinstance(trainer.optimizer, torch.optim.AdamW)

        # Cleanup
        save_dir = Path(config["training"]["save_dir"])
        if save_dir.exists():
            shutil.rmtree(save_dir)

    def test_sgd_optimizer(self, sample_config: dict, device: torch.device) -> None:
        """Test SGD optimizer creation."""
        config = sample_config.copy()
        config["training"]["optimizer"] = "sgd"
        config["model"]["segformer"]["pretrained"] = False
        config["model"]["deeplabv3plus"]["pretrained"] = False

        model = build_model(config, device)
        train_loader, val_loader = get_data_loaders(config)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        assert isinstance(trainer.optimizer, torch.optim.SGD)

        # Cleanup
        save_dir = Path(config["training"]["save_dir"])
        if save_dir.exists():
            shutil.rmtree(save_dir)

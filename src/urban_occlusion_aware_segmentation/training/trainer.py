"""Training utilities including custom loss functions and trainer class."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OcclusionWeightedLoss(nn.Module):
    """Custom loss function that emphasizes occlusion boundaries for urban segmentation.

    This loss combines cross-entropy with GPU-accelerated edge-aware weighting to penalize
    errors at object boundaries more heavily. This is critical for detecting occluded
    pedestrians and vehicles in autonomous driving scenarios where boundary precision
    directly impacts safety.

    The implementation uses efficient GPU operations (Sobel filtering and max pooling)
    instead of CPU-based numpy/scipy operations for maximum performance.

    Args:
        num_classes: Number of segmentation classes (default: 19 for Cityscapes).
        boundary_width: Width of boundary region in pixels for dilation (default: 5).
        boundary_weight: Weight multiplier for boundary pixels (default: 3.0).
            Higher values emphasize boundary accuracy more.
        ignore_index: Class index to ignore in loss computation (default: 255).

    Example:
        >>> loss_fn = OcclusionWeightedLoss(num_classes=19, boundary_weight=3.0)
        >>> predictions = torch.randn(4, 19, 512, 1024)  # (B, C, H, W)
        >>> targets = torch.randint(0, 19, (4, 512, 1024))  # (B, H, W)
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        num_classes: int = 19,
        boundary_width: int = 5,
        boundary_weight: float = 3.0,
        ignore_index: int = 255,
    ) -> None:
        """Initialize the occlusion-weighted loss function.

        Args:
            num_classes: Number of segmentation classes.
            boundary_width: Width of boundary region in pixels.
            boundary_weight: Weight multiplier for boundary pixels.
            ignore_index: Index to ignore in loss computation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute occlusion-weighted cross-entropy loss.

        Args:
            predictions: Model predictions of shape (B, C, H, W).
            targets: Ground truth labels of shape (B, H, W).

        Returns:
            Scalar loss value.
        """
        # Compute boundary masks for each sample in batch
        boundary_weights = self._compute_boundary_weights(targets)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            predictions, targets, ignore_index=self.ignore_index, reduction="none"
        )

        # Apply boundary weighting
        weighted_loss = ce_loss * boundary_weights

        # Return mean loss
        return weighted_loss.mean()

    def _compute_boundary_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel weights emphasizing boundaries using GPU operations.

        This method efficiently computes boundary weights entirely on GPU using
        PyTorch operations for maximum performance.

        Args:
            targets: Ground truth segmentation (B, H, W).

        Returns:
            Weight map of shape (B, H, W).
        """
        # Initialize weights to 1.0
        weights = torch.ones_like(targets, dtype=torch.float32, device=targets.device)

        # Compute boundaries using Sobel-like filters on GPU
        targets_float = targets.float().unsqueeze(1)  # (B, 1, H, W)

        # Sobel kernels for edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=targets.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=targets.device
        ).view(1, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(targets_float, sobel_x, padding=1)
        grad_y = F.conv2d(targets_float, sobel_y, padding=1)

        # Compute boundary mask (where gradients are non-zero)
        boundary_mask = (torch.abs(grad_x) + torch.abs(grad_y)).squeeze(1) > 0

        # Dilate boundary mask using max pooling
        if self.boundary_width > 1:
            kernel_size = self.boundary_width
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            padding = kernel_size // 2

            # Use max pooling for dilation
            boundary_mask_float = boundary_mask.float().unsqueeze(1)
            dilated = F.max_pool2d(
                boundary_mask_float,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            boundary_mask = dilated.squeeze(1) > 0.5

        # Apply boundary weights
        weights[boundary_mask] = self.boundary_weight

        return weights


class BoundaryLoss(nn.Module):
    """Auxiliary loss that focuses exclusively on boundary regions using GPU operations.

    This loss computes cross-entropy only on pixels identified as boundaries using
    Sobel edge detection. It complements the primary loss by explicitly focusing
    the model's attention on boundary precision.

    Args:
        num_classes: Number of segmentation classes (default: 19 for Cityscapes).

    Example:
        >>> loss_fn = BoundaryLoss(num_classes=19)
        >>> predictions = torch.randn(4, 19, 512, 1024)
        >>> targets = torch.randint(0, 19, (4, 512, 1024))
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, num_classes: int = 19) -> None:
        """Initialize the boundary loss function.

        Args:
            num_classes: Number of segmentation classes.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute boundary-focused loss.

        Args:
            predictions: Model predictions of shape (B, C, H, W).
            targets: Ground truth labels of shape (B, H, W).

        Returns:
            Scalar loss value.
        """
        # Extract boundaries from targets
        boundaries = self._extract_boundaries(targets)

        # Compute loss only on boundary pixels
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        boundary_loss = (ce_loss * boundaries).sum() / (boundaries.sum() + 1e-8)

        return boundary_loss

    def _extract_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        """Extract boundary mask from segmentation.

        Args:
            targets: Segmentation mask (B, H, W).

        Returns:
            Binary boundary mask (B, H, W).
        """
        # Compute gradients to find edges
        targets_float = targets.float()

        # Sobel-like filters for edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        kernel_x = kernel_x.view(1, 1, 3, 3).to(targets.device)
        kernel_y = kernel_y.view(1, 1, 3, 3).to(targets.device)

        # Apply filters
        targets_float = targets_float.unsqueeze(1)
        grad_x = F.conv2d(targets_float, kernel_x, padding=1)
        grad_y = F.conv2d(targets_float, kernel_y, padding=1)

        # Magnitude
        boundaries = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1) > 0.1

        return boundaries.float()


class Trainer:
    """Training manager for semantic segmentation models.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dictionary.
        device: Device to train on.
        mlflow_logger: Optional MLflow logger instance.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        mlflow_logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.mlflow_logger = mlflow_logger

        # Training configuration
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 50)
        self.learning_rate = train_config.get("learning_rate", 0.0001)
        self.use_amp = train_config.get("amp", True)
        self.gradient_clip = train_config.get("gradient_clip", 1.0)

        # Early stopping
        early_stopping_config = train_config.get("early_stopping", {})
        self.patience = early_stopping_config.get("patience", 10)
        self.min_delta = early_stopping_config.get("min_delta", 0.001)

        # Checkpoint configuration
        self.save_dir = Path(train_config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = train_config.get("save_freq", 5)

        # Initialize optimizer
        self.optimizer = self._build_optimizer()

        # Initialize scheduler
        self.scheduler = self._build_scheduler()

        # Initialize loss functions
        self.criterion_primary, self.criterion_auxiliary = self._build_loss()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking variables
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []

        logger.info("Trainer initialized successfully")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from configuration."""
        train_config = self.config.get("training", {})
        optimizer_name = train_config.get("optimizer", "adamw").lower()
        weight_decay = train_config.get("weight_decay", 0.0001)

        if optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Created {optimizer_name} optimizer with lr={self.learning_rate}")
        return optimizer

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        train_config = self.config.get("training", {})
        scheduler_name = train_config.get("scheduler", "cosine").lower()
        warmup_epochs = train_config.get("warmup_epochs", 5)

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Main scheduler
        if scheduler_name == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs - warmup_epochs
            )
        else:
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )

        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        logger.info(f"Created {scheduler_name} scheduler with {warmup_epochs} warmup epochs")
        return scheduler

    def _build_loss(self) -> tuple:
        """Build loss functions from configuration."""
        loss_config = self.config.get("loss", {})
        data_config = self.config.get("data", {})
        num_classes = data_config.get("num_classes", 19)

        # Primary loss
        occlusion_config = loss_config.get("occlusion", {})
        criterion_primary = OcclusionWeightedLoss(
            num_classes=num_classes,
            boundary_width=occlusion_config.get("boundary_width", 5),
            boundary_weight=occlusion_config.get("boundary_weight", 3.0),
        )

        # Auxiliary loss
        criterion_auxiliary = BoundaryLoss(num_classes=num_classes)

        logger.info("Created occlusion-weighted and boundary loss functions")
        return criterion_primary, criterion_auxiliary

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, targets)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)

                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Clear GPU cache periodically to prevent memory fragmentation
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log batch metrics
            if batch_idx % self.config.get("logging", {}).get("log_interval", 10) == 0:
                if self.mlflow_logger:
                    try:
                        self.mlflow_logger.log_metric(
                            "batch_loss", loss.item(), step=self.current_epoch * num_batches + batch_idx
                        )
                    except Exception:
                        pass

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate the model.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            outputs: Model predictions.
            targets: Ground truth labels.

        Returns:
            Weighted combination of losses.
        """
        loss_config = self.config.get("loss", {})
        weights = loss_config.get("weights", {"primary": 0.7, "auxiliary": 0.3})

        primary_loss = self.criterion_primary(outputs, targets)
        auxiliary_loss = self.criterion_auxiliary(outputs, targets)

        total_loss = (
            weights["primary"] * primary_loss + weights["auxiliary"] * auxiliary_loss
        )

        return total_loss

    def train(self) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
            )

            if self.mlflow_logger:
                try:
                    self.mlflow_logger.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "learning_rate": current_lr,
                        },
                        step=epoch,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(epoch, val_loss)

            # Check for improvement
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({self.patience} epochs without improvement)"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.2f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Current validation loss.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            checkpoint_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

#!/usr/bin/env python
"""Training script for urban occlusion-aware segmentation.

This script trains a multi-model ensemble combining SegFormer and DeepLabV3+
with custom occlusion-weighted loss for robust boundary detection in urban scenes.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from urban_occlusion_aware_segmentation.data.loader import get_data_loaders
from urban_occlusion_aware_segmentation.models.model import build_model
from urban_occlusion_aware_segmentation.training.trainer import Trainer
from urban_occlusion_aware_segmentation.utils.config import (
    get_device,
    load_config,
    save_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train urban occlusion-aware segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu/auto). Overrides config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs. Overrides config.",
    )
    return parser.parse_args()


def setup_mlflow(config: dict) -> object:
    """Setup MLflow logging.

    Args:
        config: Configuration dictionary.

    Returns:
        MLflow run object or None.
    """
    try:
        import mlflow

        mlflow_config = config.get("logging", {}).get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
        experiment_name = mlflow_config.get("experiment_name", "urban_segmentation")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run = mlflow.start_run()

        # Log configuration
        mlflow.log_params({
            "model_type": config.get("model", {}).get("type", "ensemble"),
            "learning_rate": config.get("training", {}).get("learning_rate", 0.0001),
            "batch_size": config.get("training", {}).get("batch_size", 4),
            "num_epochs": config.get("training", {}).get("num_epochs", 50),
            "optimizer": config.get("training", {}).get("optimizer", "adamw"),
        })

        logger.info(f"MLflow tracking initialized: {tracking_uri}")
        return mlflow
    except Exception as e:
        logger.warning(f"MLflow not available or failed to initialize: {e}")
        return None


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.device:
        config["system"]["device"] = args.device
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    # Setup logging
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    setup_logging(log_dir)

    # Set random seed
    seed = config.get("system", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Get device
    device_config = config.get("system", {}).get("device", "auto")
    device = get_device(device_config)

    # Setup MLflow
    mlflow_logger = setup_mlflow(config)

    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = get_data_loaders(config)
        logger.info(
            f"Data loaders created: {len(train_loader)} train batches, "
            f"{len(val_loader)} val batches"
        )

        # Build model
        logger.info("Building model...")
        model = build_model(config, device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model created: {num_params:,} total parameters, "
            f"{num_trainable:,} trainable"
        )

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            mlflow_logger=mlflow_logger,
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Save configuration
        save_dir = Path(config.get("training", {}).get("save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, str(save_dir / "config.yaml"))

        # Train model
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Best model saved to: {save_dir / 'best_model.pth'}")

        # Log final metrics to MLflow
        if mlflow_logger:
            try:
                mlflow_logger.log_metric("final_val_loss", trainer.best_val_loss)
                logger.info("Final metrics logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log final metrics to MLflow: {e}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # End MLflow run
        if mlflow_logger:
            try:
                mlflow_logger.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()

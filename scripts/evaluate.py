#!/usr/bin/env python
"""Evaluation script for urban occlusion-aware segmentation.

This script evaluates trained models on test data, computing comprehensive metrics
including mIoU, boundary F1, and occlusion recall for safety-critical scenarios.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from urban_occlusion_aware_segmentation.data.loader import get_data_loaders
from urban_occlusion_aware_segmentation.evaluation.metrics import evaluate_model
from urban_occlusion_aware_segmentation.models.model import build_model
from urban_occlusion_aware_segmentation.utils.config import (
    get_device,
    load_config,
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
        description="Evaluate urban occlusion-aware segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    # Build model architecture
    model = build_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")

    return model


def measure_inference_speed(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 50,
) -> dict:
    """Measure inference speed and FPS.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        device: Device to run on.
        num_samples: Number of samples to measure.

    Returns:
        Dictionary with timing statistics.
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_samples:
                break

            images = images.to(device)

            # Warmup
            if i < 5:
                _ = model(images)
                continue

            # Measure time
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

    # Compute statistics
    times = times[5:]  # Remove warmup
    mean_time = sum(times) / len(times)
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    return {
        "mean_inference_time_ms": mean_time * 1000,
        "inference_fps": fps,
        "num_samples_measured": len(times),
    }


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.device:
        config["system"]["device"] = args.device
    if args.batch_size:
        config["evaluation"]["batch_size"] = args.batch_size

    # Setup logging
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    setup_logging(log_dir)

    # Set random seed
    seed = config.get("system", {}).get("seed", 42)
    set_seed(seed)

    # Get device
    device_config = config.get("system", {}).get("device", "auto")
    device = get_device(device_config)

    try:
        # Check checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        # Load model
        logger.info("Loading model...")
        model = load_model(str(checkpoint_path), config, device)

        # Create data loaders
        logger.info("Creating data loaders...")
        _, val_loader = get_data_loaders(config)

        # Evaluate model
        logger.info("Evaluating model...")
        eval_config = config.get("evaluation", {})
        num_classes = config.get("data", {}).get("num_classes", 19)
        occlusion_classes = eval_config.get("occlusion_classes", [11, 12, 13, 14])

        metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
            occlusion_classes=occlusion_classes,
        )

        # Measure inference speed
        logger.info("Measuring inference speed...")
        timing_stats = measure_inference_speed(model, val_loader, device)
        metrics.update(timing_stats)

        # Log results
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        logger.info("=" * 50)

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "checkpoint": str(checkpoint_path),
            "config": str(args.config),
            "metrics": metrics,
            "evaluation_settings": {
                "num_classes": num_classes,
                "occlusion_classes": occlusion_classes,
                "device": str(device),
            },
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

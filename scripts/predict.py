#!/usr/bin/env python
"""Inference script for urban occlusion-aware segmentation.

This script performs inference on images using trained models, producing
segmentation masks and uncertainty maps.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from urban_occlusion_aware_segmentation.models.model import build_model
from urban_occlusion_aware_segmentation.utils.config import (
    get_device,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


# Cityscapes color palette for visualization
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with urban occlusion-aware segmentation model"
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
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        help="Save uncertainty maps (for ensemble models)",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlays of predictions on input images",
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


def preprocess_image(
    image_path: str, target_size: tuple = (256, 512)
) -> tuple[torch.Tensor, tuple, np.ndarray]:
    """Load and preprocess image for inference.

    Args:
        image_path: Path to input image.
        target_size: Target size (H, W) for model input.

    Returns:
        Preprocessed tensor, original size, and original image.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_size = image.shape[:2]
    original_image = image.copy()

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (target_size[1], target_size[0]))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image / 255.0 - mean) / std

    # Convert to tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

    return tensor, original_size, original_image


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class indices to RGB colors.

    Args:
        mask: Segmentation mask with class indices (H, W).

    Returns:
        RGB colored mask (H, W, 3).
    """
    colored = CITYSCAPES_COLORS[mask]
    return colored


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of segmentation mask on original image.

    Args:
        image: Original image (H, W, 3) in BGR.
        mask: Colored segmentation mask (H, W, 3) in RGB.
        alpha: Blending factor.

    Returns:
        Blended overlay (H, W, 3) in BGR.
    """
    # Convert mask to BGR
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    # Resize mask to match image size
    if mask_bgr.shape[:2] != image.shape[:2]:
        mask_bgr = cv2.resize(
            mask_bgr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, mask_bgr, alpha, 0)

    return overlay


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    target_size: tuple = (256, 512),
    save_uncertainty: bool = False,
) -> dict:
    """Run inference on a single image.

    Args:
        model: Trained model.
        image_path: Path to input image.
        device: Device to run on.
        target_size: Target input size for model.
        save_uncertainty: Whether to compute uncertainty (ensemble only).

    Returns:
        Dictionary with predictions and metadata.
    """
    # Preprocess
    tensor, original_size, original_image = preprocess_image(image_path, target_size)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        if save_uncertainty and hasattr(model, "forward"):
            try:
                output, uncertainty = model(tensor, return_uncertainty=True)
            except TypeError:
                # Model doesn't support uncertainty
                output = model(tensor)
                uncertainty = None
        else:
            output = model(tensor)
            uncertainty = None

    # Get predictions
    predictions = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Resize to original size
    predictions = cv2.resize(
        predictions.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Process uncertainty if available
    if uncertainty is not None:
        uncertainty_map = uncertainty.squeeze(0).cpu().numpy()
        uncertainty_map = cv2.resize(
            uncertainty_map, (original_size[1], original_size[0])
        )
    else:
        uncertainty_map = None

    return {
        "predictions": predictions,
        "uncertainty": uncertainty_map,
        "original_image": original_image,
        "original_size": original_size,
    }


def main() -> None:
    """Main inference function."""
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

    # Setup logging
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    setup_logging(log_dir)

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

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get target size from config
        target_size = tuple(config.get("data", {}).get("image_size", [256, 512]))

        # Process input
        input_path = Path(args.input)

        if input_path.is_file():
            # Single image
            image_paths = [input_path]
        elif input_path.is_dir():
            # Directory of images
            image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        else:
            logger.error(f"Invalid input path: {args.input}")
            sys.exit(1)

        logger.info(f"Processing {len(image_paths)} image(s)...")

        # Process each image
        for img_path in image_paths:
            logger.info(f"Processing {img_path.name}...")

            # Run inference
            results = predict_image(
                model, str(img_path), device, target_size, args.save_uncertainty
            )

            # Save segmentation mask (colored)
            colored_mask = colorize_mask(results["predictions"])
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved segmentation mask to {mask_path}")

            # Save overlay if requested
            if args.save_overlay:
                overlay = create_overlay(results["original_image"], colored_mask)
                overlay_path = output_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                logger.info(f"Saved overlay to {overlay_path}")

            # Save uncertainty map if available
            if args.save_uncertainty and results["uncertainty"] is not None:
                uncertainty_vis = (results["uncertainty"] * 255).astype(np.uint8)
                uncertainty_vis = cv2.applyColorMap(uncertainty_vis, cv2.COLORMAP_JET)
                uncertainty_path = output_dir / f"{img_path.stem}_uncertainty.png"
                cv2.imwrite(str(uncertainty_path), uncertainty_vis)
                logger.info(f"Saved uncertainty map to {uncertainty_path}")

        logger.info(f"\nInference completed! Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Inference failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Data preprocessing and augmentation pipelines."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Optional imports with fallback
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ToTensorV2 = None

logger = logging.getLogger(__name__)


def get_train_transforms(config: Dict[str, Any]) -> Optional[Union[A.Compose, Any]]:
    """Create training data augmentation pipeline.

    Args:
        config: Configuration dictionary containing augmentation parameters.

    Returns:
        Albumentations composition of transformations, or None if albumentations unavailable.

    Raises:
        ImportError: If albumentations is not available.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install with: pip install albumentations"
        )

    aug_config = config.get("augmentation", {})
    train_config = aug_config.get("train", {})
    norm_config = aug_config.get("normalize", {})

    image_size = config.get("data", {}).get("image_size", [512, 1024])

    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=train_config.get("horizontal_flip", 0.5)),
        A.Rotate(
            limit=train_config.get("random_rotate", 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
    ]

    # Color augmentation
    color_jitter = train_config.get("color_jitter", {})
    if color_jitter:
        transforms.append(
            A.ColorJitter(
                brightness=color_jitter.get("brightness", 0.3),
                contrast=color_jitter.get("contrast", 0.3),
                saturation=color_jitter.get("saturation", 0.3),
                hue=color_jitter.get("hue", 0.1),
                p=0.5,
            )
        )

    # Gaussian blur
    if train_config.get("gaussian_blur", 0.0) > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=train_config["gaussian_blur"]))

    # Normalization
    transforms.extend([
        A.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_val_transforms(config: Dict[str, Any]) -> Optional[Union[A.Compose, Any]]:
    """Create validation data transformation pipeline.

    Args:
        config: Configuration dictionary containing augmentation parameters.

    Returns:
        Albumentations composition of transformations, or None if albumentations unavailable.

    Raises:
        ImportError: If albumentations is not available.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install with: pip install albumentations"
        )

    aug_config = config.get("augmentation", {})
    val_config = aug_config.get("val", {})
    norm_config = aug_config.get("normalize", {})

    image_size = val_config.get("resize", [512, 1024])

    transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=norm_config.get("mean", [0.485, 0.456, 0.406]),
            std=norm_config.get("std", [0.229, 0.224, 0.225]),
        ),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def compute_boundary_mask(segmentation: np.ndarray, width: int = 5) -> np.ndarray:
    """Compute boundary mask from segmentation map using morphological operations.

    Args:
        segmentation: Segmentation mask as numpy array (H, W).
        width: Boundary width in pixels.

    Returns:
        Binary boundary mask of same shape as input.
    """
    # Dilate and erode to find boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
    dilated = cv2.dilate(segmentation.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(segmentation.astype(np.uint8), kernel, iterations=1)

    # Boundary is difference between dilated and eroded
    boundary = (dilated != eroded).astype(np.float32)

    return boundary


def compute_edge_weights(
    segmentation: np.ndarray,
    boundary_width: int = 5,
    boundary_weight: float = 3.0,
    edge_threshold: float = 0.1,
) -> np.ndarray:
    """Compute edge-aware weights for occlusion-weighted loss.

    This function creates a weight map that emphasizes object boundaries and
    occlusion regions, which are critical for autonomous vehicle perception.

    Args:
        segmentation: Ground truth segmentation mask (H, W).
        boundary_width: Width of boundary region in pixels.
        boundary_weight: Weight multiplier for boundary pixels.
        edge_threshold: Threshold for edge detection.

    Returns:
        Weight map of same shape as segmentation.
    """
    # Initialize weights to 1
    weights = np.ones_like(segmentation, dtype=np.float32)

    # Compute boundaries
    boundary_mask = compute_boundary_mask(segmentation, boundary_width)

    # Apply higher weights to boundary regions
    weights[boundary_mask > 0] = boundary_weight

    return weights


def extract_occlusion_regions(
    segmentation: np.ndarray, occlusion_classes: List[int]
) -> np.ndarray:
    """Extract regions corresponding to potentially occluded objects.

    Args:
        segmentation: Segmentation mask (H, W).
        occlusion_classes: List of class IDs considered for occlusion analysis
                          (e.g., person, rider, car, truck).

    Returns:
        Binary mask indicating potential occlusion regions.
    """
    occlusion_mask = np.zeros_like(segmentation, dtype=np.float32)

    for cls_id in occlusion_classes:
        occlusion_mask[segmentation == cls_id] = 1.0

    return occlusion_mask


def normalize_tensor(
    tensor: np.ndarray, mean: List[float], std: List[float]
) -> np.ndarray:
    """Normalize tensor with given mean and std.

    Args:
        tensor: Input tensor of shape (C, H, W) or (H, W, C).
        mean: Mean values for each channel.
        std: Standard deviation values for each channel.

    Returns:
        Normalized tensor.
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    return (tensor - mean) / std


def denormalize_tensor(
    tensor: np.ndarray, mean: List[float], std: List[float]
) -> np.ndarray:
    """Denormalize tensor for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W).
        mean: Mean values used for normalization.
        std: Standard deviation values used for normalization.

    Returns:
        Denormalized tensor.
    """
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)

    return tensor * std + mean

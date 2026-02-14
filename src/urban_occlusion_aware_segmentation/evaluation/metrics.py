"""Evaluation metrics for semantic segmentation with occlusion awareness."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def compute_miou(
    predictions: np.ndarray, targets: np.ndarray, num_classes: int, ignore_index: int = 255
) -> float:
    """Compute mean Intersection over Union (mIoU).

    Args:
        predictions: Predicted segmentation masks (H, W) or (B, H, W).
        targets: Ground truth masks (H, W) or (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        Mean IoU across all classes.
    """
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Remove ignored pixels
    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))

    # Compute IoU for each class
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-10)

    # Mean IoU (excluding classes with no samples)
    valid_classes = union > 0
    miou = iou[valid_classes].mean()

    return float(miou)


def compute_boundary_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    boundary_threshold: int = 5,
    ignore_index: int = 255,
) -> float:
    """Compute F1 score for boundary pixels.

    This metric specifically evaluates prediction quality at object boundaries,
    which is crucial for detecting occlusions.

    Args:
        predictions: Predicted segmentation masks (H, W) or (B, H, W).
        targets: Ground truth masks (H, W) or (B, H, W).
        boundary_threshold: Distance threshold in pixels for boundary matching.
        ignore_index: Index to ignore in computation.

    Returns:
        F1 score for boundary regions.
    """
    if predictions.ndim == 3:
        # Batch processing
        f1_scores = []
        for pred, target in zip(predictions, targets):
            f1_scores.append(
                compute_boundary_f1(pred, target, boundary_threshold, ignore_index)
            )
        return float(np.mean(f1_scores))

    # Extract boundaries
    pred_boundary = _extract_boundary(predictions)
    target_boundary = _extract_boundary(targets)

    # Compute distance transforms for matching
    if target_boundary.sum() == 0:
        return 1.0 if pred_boundary.sum() == 0 else 0.0

    target_dist = distance_transform_edt(1 - target_boundary)

    # True positives: predicted boundary pixels close to target boundaries
    tp = np.sum((pred_boundary > 0) & (target_dist <= boundary_threshold))

    # False positives: predicted boundary pixels far from target boundaries
    fp = np.sum((pred_boundary > 0) & (target_dist > boundary_threshold))

    # False negatives: target boundary pixels not matched
    pred_dist = distance_transform_edt(1 - pred_boundary)
    fn = np.sum((target_boundary > 0) & (pred_dist > boundary_threshold))

    # Compute F1 score
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return float(f1)


def compute_occlusion_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    occlusion_classes: List[int],
    boundary_width: int = 5,
) -> float:
    """Compute recall specifically for potentially occluded objects.

    This metric focuses on critical safety-relevant classes (pedestrians, vehicles)
    near occlusion boundaries.

    Args:
        predictions: Predicted segmentation masks (H, W) or (B, H, W).
        targets: Ground truth masks (H, W) or (B, H, W).
        occlusion_classes: List of class IDs to consider (e.g., person, car).
        boundary_width: Width around boundaries to consider as occlusion regions.

    Returns:
        Recall for occlusion regions.
    """
    if predictions.ndim == 3:
        # Batch processing
        recalls = []
        for pred, target in zip(predictions, targets):
            recalls.append(
                compute_occlusion_recall(pred, target, occlusion_classes, boundary_width)
            )
        return float(np.mean(recalls))

    # Extract boundary regions
    boundary_mask = _extract_boundary(targets)

    # Dilate boundary to create occlusion region
    from scipy.ndimage import binary_dilation

    occlusion_region = binary_dilation(boundary_mask, iterations=boundary_width // 2)

    # Mask for occlusion-prone classes
    occlusion_class_mask = np.isin(targets, occlusion_classes)

    # Combined mask: occlusion region AND occlusion-prone classes
    occlusion_mask = occlusion_region & occlusion_class_mask

    if occlusion_mask.sum() == 0:
        return 1.0

    # Compute recall in occlusion regions
    correct = (predictions == targets) & occlusion_mask
    recall = correct.sum() / occlusion_mask.sum()

    return float(recall)


def compute_pixel_accuracy(
    predictions: np.ndarray, targets: np.ndarray, ignore_index: int = 255
) -> float:
    """Compute overall pixel accuracy.

    Args:
        predictions: Predicted segmentation masks.
        targets: Ground truth masks.
        ignore_index: Index to ignore in computation.

    Returns:
        Pixel accuracy.
    """
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Remove ignored pixels
    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    accuracy = (predictions == targets).sum() / len(targets)
    return float(accuracy)


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels from segmentation mask.

    Args:
        mask: Segmentation mask (H, W).

    Returns:
        Binary boundary mask.
    """
    # Compute gradients
    grad_x = np.abs(np.gradient(mask.astype(float), axis=0))
    grad_y = np.abs(np.gradient(mask.astype(float), axis=1))

    # Boundary where gradient magnitude is non-zero
    boundary = (grad_x + grad_y) > 0

    return boundary.astype(np.uint8)


class SegmentationMetrics:
    """Comprehensive metrics tracker for semantic segmentation.

    Args:
        num_classes: Number of segmentation classes.
        occlusion_classes: List of class IDs for occlusion analysis.
        boundary_threshold: Distance threshold for boundary F1 computation.
        ignore_index: Index to ignore in metrics computation.
    """

    def __init__(
        self,
        num_classes: int = 19,
        occlusion_classes: Optional[List[int]] = None,
        boundary_threshold: int = 5,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = num_classes
        self.occlusion_classes = occlusion_classes or [11, 12, 13, 14]
        self.boundary_threshold = boundary_threshold
        self.ignore_index = ignore_index

        # Accumulated predictions and targets
        self.all_predictions: List[np.ndarray] = []
        self.all_targets: List[np.ndarray] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with new batch.

        Args:
            predictions: Model predictions (B, C, H, W) or (B, H, W).
            targets: Ground truth labels (B, H, W).
        """
        # Convert logits to predictions if needed
        if predictions.ndim == 4:
            predictions = predictions.argmax(dim=1)

        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()

        self.all_predictions.append(predictions)
        self.all_targets.append(targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names to values.
        """
        if not self.all_predictions:
            logger.warning("No predictions to compute metrics")
            return {}

        # Concatenate all batches
        predictions = np.concatenate(self.all_predictions, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)

        # Compute metrics
        metrics = {
            "miou": compute_miou(predictions, targets, self.num_classes, self.ignore_index),
            "pixel_accuracy": compute_pixel_accuracy(predictions, targets, self.ignore_index),
            "boundary_f1": compute_boundary_f1(
                predictions, targets, self.boundary_threshold, self.ignore_index
            ),
            "occlusion_recall": compute_occlusion_recall(
                predictions, targets, self.occlusion_classes
            ),
        }

        logger.info("Computed metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def reset(self) -> None:
        """Reset accumulated predictions and targets."""
        self.all_predictions = []
        self.all_targets = []


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 19,
    occlusion_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.
        num_classes: Number of segmentation classes.
        occlusion_classes: List of occlusion-prone class IDs.

    Returns:
        Dictionary of metric results.
    """
    model.eval()
    metrics_tracker = SegmentationMetrics(
        num_classes=num_classes, occlusion_classes=occlusion_classes
    )

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Update metrics
            metrics_tracker.update(outputs, targets)

    # Compute final metrics
    results = metrics_tracker.compute()

    return results

"""Model implementations for urban occlusion-aware segmentation.

This module implements transformer-based (SegFormer) and CNN-based (DeepLabV3+)
architectures, along with an ensemble model that combines both with uncertainty
quantification for robust occlusion handling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101

# Optional imports with fallback
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None

try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SegFormerSegmentation(nn.Module):
    """SegFormer-based semantic segmentation model.

    SegFormer uses a hierarchical transformer encoder and lightweight MLP decoder
    for efficient and accurate segmentation. This implementation attempts to use
    the HuggingFace transformers library for proper SegFormer, falling back to
    timm-based implementation if unavailable.

    Args:
        num_classes: Number of segmentation classes.
        backbone: SegFormer backbone variant ('mit_b0' to 'mit_b5' or 'nvidia/mit-b3').
        pretrained: Whether to use pretrained weights.
        dropout: Dropout rate for the decoder.
    """

    def __init__(
        self,
        num_classes: int = 19,
        backbone: str = "mit_b3",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_transformers = False

        # Try to use HuggingFace transformers SegFormer first (proper implementation)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Map backbone names to HuggingFace model IDs
                backbone_map = {
                    "mit_b0": "nvidia/mit-b0",
                    "mit_b1": "nvidia/mit-b1",
                    "mit_b2": "nvidia/mit-b2",
                    "mit_b3": "nvidia/mit-b3",
                    "mit_b4": "nvidia/mit-b4",
                    "mit_b5": "nvidia/mit-b5",
                }

                model_name = backbone_map.get(backbone, backbone)

                if pretrained:
                    try:
                        # Load pretrained SegFormer model
                        self.model = SegformerForSemanticSegmentation.from_pretrained(
                            model_name,
                            num_labels=num_classes,
                            ignore_mismatched_sizes=True,
                        )
                        self.use_transformers = True
                        logger.info(f"Loaded pretrained SegFormer {model_name} from HuggingFace")
                    except Exception as e:
                        logger.warning(f"Failed to load pretrained model: {e}, using random init")
                        raise RuntimeError("Pretrained model unavailable")
                else:
                    # Create model with random initialization
                    config = SegformerConfig.from_pretrained(model_name)
                    config.num_labels = num_classes
                    self.model = SegformerForSemanticSegmentation(config)
                    self.use_transformers = True
                    logger.info(f"Initialized SegFormer {model_name} with random weights")

            except (ImportError, RuntimeError, Exception) as e:
                logger.warning(
                    f"HuggingFace transformers SegFormer not available: {e}. "
                    f"Falling back to timm-based implementation"
                )
                raise RuntimeError("Transformers SegFormer unavailable")
        else:
            logger.warning(
                "HuggingFace transformers not available. "
                "Falling back to timm-based implementation"
            )

        # Fallback to timm-based implementation
        if not self.use_transformers:
            if not TIMM_AVAILABLE:
                raise ImportError(
                    "Neither transformers nor timm are available. "
                    "Please install at least one: pip install timm or pip install transformers"
                )

            try:
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                )
                logger.info(f"Loaded {backbone} backbone from timm")
            except (RuntimeError, Exception):
                # Final fallback to ResNet-based feature extractor
                logger.warning(f"{backbone} not available in timm, using ResNet50 as fallback")
                self.backbone = timm.create_model(
                    "resnet50",
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                )

            # Get feature dimensions with smaller dummy input to save memory
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 256, 256)
                features = self.backbone(dummy)
                self.feature_dims = [f.shape[1] for f in features]
                del dummy, features  # Free memory immediately
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # MLP decoder head
            self.decode_head = MLPDecoder(
                in_channels=self.feature_dims,
                embedding_dim=256,
                num_classes=num_classes,
                dropout=dropout,
            )

            logger.info(
                f"Initialized SegFormer fallback with {num_classes} classes, "
                f"feature dims: {self.feature_dims}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        if self.use_transformers:
            # Use HuggingFace transformers SegFormer
            outputs = self.model(pixel_values=x)
            logits = outputs.logits

            # Upsample to input resolution
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            return logits
        else:
            # Use timm-based fallback implementation
            # Extract multi-scale features
            features = self.backbone(x)

            # Decode to segmentation map
            output = self.decode_head(features)

            # Upsample to input resolution
            output = F.interpolate(
                output, size=x.shape[2:], mode="bilinear", align_corners=False
            )

            return output


class MLPDecoder(nn.Module):
    """Lightweight MLP decoder for SegFormer.

    Args:
        in_channels: List of input channel dimensions for each scale.
        embedding_dim: Unified embedding dimension.
        num_classes: Number of output classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: List[int],
        embedding_dim: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project each feature scale to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embedding_dim, 1),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True),
            )
            for dim in in_channels
        ])

        # Fusion and segmentation head
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.segmentation_head = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature tensors from different scales.

        Returns:
            Segmentation logits.
        """
        # Get target size (smallest feature map upsampled 4x)
        # Limit target size to prevent excessive memory usage
        base_h, base_w = features[0].shape[2], features[0].shape[3]
        target_size = (min(base_h * 4, 128), min(base_w * 4, 256))

        # Project and upsample all features to same size
        projected = []
        for feat, proj in zip(features, self.projections):
            feat = proj(feat)
            # Use memory-efficient interpolation
            with torch.cuda.amp.autocast(enabled=False):
                feat = F.interpolate(
                    feat.float(),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )
            projected.append(feat)

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)
        fused = self.fusion(fused)

        # Generate segmentation
        output = self.segmentation_head(fused)

        return output


class DeepLabV3PlusSegmentation(nn.Module):
    """DeepLabV3+ semantic segmentation model with ResNet backbone.

    DeepLabV3+ combines atrous spatial pyramid pooling (ASPP) with an encoder-decoder
    structure for capturing multi-scale context.

    Args:
        num_classes: Number of segmentation classes.
        backbone: Backbone architecture ('resnet50' or 'resnet101').
        pretrained: Whether to use pretrained weights.
        output_stride: Output stride for atrous convolutions (8 or 16).
    """

    def __init__(
        self,
        num_classes: int = 19,
        backbone: str = "resnet101",
        pretrained: bool = True,
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Use torchvision's DeepLabV3 as base
        if backbone == "resnet101":
            base_model = deeplabv3_resnet101(pretrained=pretrained)
        else:
            from torchvision.models.segmentation import deeplabv3_resnet50
            base_model = deeplabv3_resnet50(pretrained=pretrained)

        self.backbone = base_model.backbone
        self.aspp = base_model.classifier[0]

        # Replace classifier head with correct number of classes
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        logger.info(
            f"Initialized DeepLabV3+ with {num_classes} classes, "
            f"backbone: {backbone}, output_stride: {output_stride}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        input_shape = x.shape[-2:]

        # Extract features
        features = self.backbone(x)
        x = features["out"]

        # Apply ASPP
        x = self.aspp(x)

        # Classification
        x = self.classifier(x)

        # Upsample to input resolution
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


class OcclusionAwareEnsemble(nn.Module):
    """Ensemble model combining SegFormer and DeepLabV3+ with uncertainty quantification.

    This ensemble explicitly handles occlusion boundaries by:
    1. Computing predictions from both models
    2. Quantifying prediction uncertainty (disagreement between models)
    3. Using uncertainty to weight the ensemble, emphasizing agreement regions

    Args:
        num_classes: Number of segmentation classes.
        segformer_config: Configuration dict for SegFormer model.
        deeplabv3_config: Configuration dict for DeepLabV3+ model.
        ensemble_weights: Weights for combining models [w_segformer, w_deeplabv3].
        uncertainty_threshold: Threshold for high uncertainty regions.
    """

    def __init__(
        self,
        num_classes: int = 19,
        segformer_config: Optional[Dict] = None,
        deeplabv3_config: Optional[Dict] = None,
        ensemble_weights: List[float] = [0.5, 0.5],
        uncertainty_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ensemble_weights = torch.tensor(ensemble_weights)
        self.uncertainty_threshold = uncertainty_threshold

        # Initialize models
        segformer_config = segformer_config or {}
        deeplabv3_config = deeplabv3_config or {}

        self.segformer = SegFormerSegmentation(
            num_classes=num_classes,
            backbone=segformer_config.get("backbone", "mit_b3"),
            pretrained=segformer_config.get("pretrained", True),
            dropout=segformer_config.get("dropout", 0.1),
        )

        self.deeplabv3 = DeepLabV3PlusSegmentation(
            num_classes=num_classes,
            backbone=deeplabv3_config.get("backbone", "resnet101"),
            pretrained=deeplabv3_config.get("pretrained", True),
            output_stride=deeplabv3_config.get("output_stride", 16),
        )

        logger.info(
            f"Initialized OcclusionAwareEnsemble with {num_classes} classes, "
            f"weights: {ensemble_weights}, uncertainty_threshold: {uncertainty_threshold}"
        )

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional uncertainty map.

        Args:
            x: Input tensor of shape (B, 3, H, W).
            return_uncertainty: Whether to return uncertainty map.

        Returns:
            Segmentation logits, and optionally uncertainty map.
        """
        # Get predictions from both models sequentially to save memory
        pred_segformer = self.segformer(x)

        # Clear cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pred_deeplabv3 = self.deeplabv3(x)

        # Compute uncertainty as disagreement between models
        uncertainty = self._compute_uncertainty(pred_segformer, pred_deeplabv3)

        # Ensemble with uncertainty-aware weighting
        weights = self.ensemble_weights.to(x.device)

        # Adaptive weighting based on uncertainty
        # High uncertainty -> equal weighting; Low uncertainty -> use specified weights
        adaptive_weight = torch.sigmoid((self.uncertainty_threshold - uncertainty) * 10)
        adaptive_weight = adaptive_weight.unsqueeze(1)  # (B, 1, H, W)

        # Weighted combination
        output = (
            weights[0] * pred_segformer + weights[1] * pred_deeplabv3
        ) * adaptive_weight + (
            0.5 * pred_segformer + 0.5 * pred_deeplabv3
        ) * (1 - adaptive_weight)

        if return_uncertainty:
            return output, uncertainty
        return output

    def _compute_uncertainty(
        self, pred1: torch.Tensor, pred2: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty as model disagreement.

        Args:
            pred1: Predictions from first model (B, C, H, W).
            pred2: Predictions from second model (B, C, H, W).

        Returns:
            Uncertainty map (B, H, W) in range [0, 1].
        """
        # Convert logits to probabilities
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)

        # Compute disagreement as Jensen-Shannon divergence
        mean_prob = (prob1 + prob2) / 2
        kl1 = F.kl_div(
            mean_prob.log(), prob1, reduction="none", log_target=False
        ).sum(dim=1)
        kl2 = F.kl_div(
            mean_prob.log(), prob2, reduction="none", log_target=False
        ).sum(dim=1)

        # JS divergence
        js_div = (kl1 + kl2) / 2

        # Normalize to [0, 1]
        uncertainty = js_div / (js_div.max() + 1e-8)

        return uncertainty


def build_model(config: Dict, device: torch.device) -> nn.Module:
    """Build model from configuration with comprehensive error handling.

    Args:
        config: Configuration dictionary containing model specifications.
        device: Device to place model on (cuda/cpu).

    Returns:
        Initialized model ready for training or inference.

    Raises:
        ValueError: If model type is unknown or configuration is invalid.
        RuntimeError: If model initialization fails.
    """
    try:
        model_config = config.get("model", {})
        data_config = config.get("data", {})

        num_classes = data_config.get("num_classes", 19)
        model_type = model_config.get("type", "ensemble")

        logger.info(f"Building {model_type} model with {num_classes} classes")

        if model_type == "segformer":
            model = SegFormerSegmentation(
                num_classes=num_classes,
                backbone=model_config.get("segformer", {}).get("backbone", "mit_b3"),
                pretrained=model_config.get("segformer", {}).get("pretrained", True),
                dropout=model_config.get("segformer", {}).get("dropout", 0.1),
            )
        elif model_type == "deeplabv3plus":
            model = DeepLabV3PlusSegmentation(
                num_classes=num_classes,
                backbone=model_config.get("deeplabv3plus", {}).get("backbone", "resnet101"),
                pretrained=model_config.get("deeplabv3plus", {}).get("pretrained", True),
                output_stride=model_config.get("deeplabv3plus", {}).get("output_stride", 16),
            )
        elif model_type == "ensemble":
            model = OcclusionAwareEnsemble(
                num_classes=num_classes,
                segformer_config=model_config.get("segformer", {}),
                deeplabv3_config=model_config.get("deeplabv3plus", {}),
                ensemble_weights=model_config.get("ensemble", {}).get("weights", [0.5, 0.5]),
                uncertainty_threshold=model_config.get("ensemble", {}).get(
                    "uncertainty_threshold", 0.3
                ),
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: segformer, deeplabv3plus, ensemble"
            )

        # Move model to device with error handling
        try:
            model = model.to(device)
            logger.info(f"Successfully built and moved {model_type} model to {device}")
        except RuntimeError as e:
            logger.error(f"Failed to move model to {device}: {e}")
            raise RuntimeError(
                f"Could not allocate model on {device}. "
                f"Try using CPU or reducing model size."
            ) from e

        return model

    except ValueError as e:
        logger.error(f"Configuration error while building model: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while building model: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failed: {e}") from e

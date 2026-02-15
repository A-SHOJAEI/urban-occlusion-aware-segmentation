"""Custom neural network components for occlusion-aware segmentation.

This module implements specialized components including attention mechanisms,
boundary refinement modules, and adaptive fusion layers for improved
occlusion handling in semantic segmentation.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BoundaryAttention(nn.Module):
    """Boundary attention module for emphasizing edge regions.

    This module learns to attend to boundary regions by computing spatial
    attention weights based on local feature gradients and context.

    Args:
        in_channels: Number of input channels.
        reduction: Channel reduction ratio for attention computation.
    """

    def __init__(self, in_channels: int, reduction: int = 8) -> None:
        super().__init__()

        self.in_channels = in_channels

        # Spatial attention pathway
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        # Edge detection pathway (learnable Sobel-like filters)
        self.edge_conv_x = nn.Conv2d(
            in_channels, 1, kernel_size=3, padding=1, bias=False
        )
        self.edge_conv_y = nn.Conv2d(
            in_channels, 1, kernel_size=3, padding=1, bias=False
        )

        # Initialize with Sobel-like filters
        self._init_edge_filters()

        logger.debug(f"Initialized BoundaryAttention with {in_channels} channels")

    def _init_edge_filters(self) -> None:
        """Initialize edge detection filters with Sobel-like patterns."""
        # Sobel x-direction
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel y-direction
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Replicate across input channels
        self.edge_conv_x.weight.data = sobel_x.repeat(self.in_channels, 1, 1, 1) / self.in_channels
        self.edge_conv_y.weight.data = sobel_y.repeat(self.in_channels, 1, 1, 1) / self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor (B, C, H, W).

        Returns:
            Attention-weighted features (B, C, H, W).
        """
        # Compute spatial attention
        spatial_att = self.spatial_conv(x)

        # Compute edge magnitude
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge_att = torch.sigmoid(edge_magnitude)

        # Combine attentions
        combined_att = spatial_att * (1 + edge_att)  # Boost boundary regions

        # Apply attention
        out = x * combined_att

        return out


class AdaptiveFusionModule(nn.Module):
    """Adaptive fusion module for combining multi-scale features.

    This module learns to dynamically weight and fuse features from different
    scales based on their relevance to the current spatial location.

    Args:
        in_channels: List of input channel dimensions for each scale.
        out_channels: Output channel dimension.
        num_scales: Number of input scales.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_scales: Optional[int] = None
    ) -> None:
        super().__init__()

        if num_scales is None:
            num_scales = len(in_channels)

        self.num_scales = num_scales

        # Per-scale projection layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels
        ])

        # Attention weights for adaptive fusion
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * num_scales, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_scales, 1),
            nn.Softmax(dim=1)  # Attention weights across scales
        )

        logger.debug(
            f"Initialized AdaptiveFusionModule with {num_scales} scales, "
            f"output channels: {out_channels}"
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature tensors from different scales.

        Returns:
            Fused feature tensor (B, out_channels, H, W).
        """
        # Get target size from first feature
        target_size = features[0].shape[2:]

        # Project and resize all features
        projected = []
        for feat, proj in zip(features, self.projections):
            feat = proj(feat)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            projected.append(feat)

        # Compute attention weights
        concat_features = torch.cat(projected, dim=1)
        attention_weights = self.attention(concat_features)  # (B, num_scales, H, W)

        # Weighted fusion
        fused = sum(
            w.unsqueeze(1) * f
            for w, f in zip(attention_weights.split(1, dim=1), projected)
        )

        return fused


class OcclusionRefinementModule(nn.Module):
    """Refinement module specifically designed for occlusion boundaries.

    This module refines segmentation predictions by explicitly modeling
    occlusion relationships and boundary contexts.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of segmentation classes.
        hidden_channels: Number of hidden channels.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128
    ) -> None:
        super().__init__()

        # Boundary detection branch
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()  # Boundary probability
        )

        # Context aggregation for boundary regions
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Refinement head
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )

        logger.debug(
            f"Initialized OcclusionRefinementModule for {num_classes} classes"
        )

    def forward(
        self, features: torch.Tensor, coarse_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input feature tensor (B, C, H, W).
            coarse_pred: Coarse predictions (B, num_classes, H, W).

        Returns:
            Tuple of (refined predictions, boundary map).
        """
        # Detect boundaries
        boundary_map = self.boundary_branch(features)

        # Aggregate context for boundary regions
        context = self.context_conv(features)

        # Combine features and context, weighted by boundaries
        context_weighted = context * boundary_map
        combined = torch.cat([features, context_weighted], dim=1)

        # Refine predictions
        refinement_delta = self.refinement(combined)

        # Add residual connection
        refined_pred = coarse_pred + refinement_delta

        return refined_pred, boundary_map


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration.

    Implements squeeze-and-excitation style channel attention to
    emphasize informative feature channels.

    Args:
        in_channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Attention-weighted tensor (B, C, H, W).
        """
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        out = x * attention

        return out


class PyramidPoolingModule(nn.Module):
    """Pyramid pooling module for multi-scale context aggregation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        pool_sizes: List of pooling sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: list[int] = [1, 2, 3, 6]
    ) -> None:
        super().__init__()

        self.pool_sizes = pool_sizes

        # Per-level convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1),
                nn.BatchNorm2d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Multi-scale pooled features (B, out_channels, H, W).
        """
        input_size = x.shape[2:]

        # Apply pyramid pooling
        pooled = []
        for conv in self.convs:
            feat = conv(x)
            feat = F.interpolate(
                feat, size=input_size, mode='bilinear', align_corners=False
            )
            pooled.append(feat)

        # Concatenate all levels with input
        concat = torch.cat([x] + pooled, dim=1)

        # Fuse
        out = self.fusion(concat)

        return out

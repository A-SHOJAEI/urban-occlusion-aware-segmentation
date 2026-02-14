"""Urban Occlusion-Aware Segmentation.

Multi-model ensemble for semantic segmentation with explicit occlusion boundary handling
in urban scenes using uncertainty quantification.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from urban_occlusion_aware_segmentation.models.model import (
    OcclusionAwareEnsemble,
    SegFormerSegmentation,
    DeepLabV3PlusSegmentation,
)

__all__ = [
    "OcclusionAwareEnsemble",
    "SegFormerSegmentation",
    "DeepLabV3PlusSegmentation",
]

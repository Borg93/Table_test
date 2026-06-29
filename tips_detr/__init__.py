"""TipsDETR — frozen TIPS v2 backbone + multi-scale deformable DETR (Exp B).

Our own detector (Apache-clean, not an RF-DETR fork) for handwritten table
structure, built for offline/batch inference. See tips_detr/README.md.
"""
from .model import TipsDETR, TipsDETRConfig, post_process

__all__ = ["TipsDETR", "TipsDETRConfig", "post_process"]

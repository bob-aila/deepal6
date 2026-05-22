"""
deepal.models
-------------
Model architecture re-exports for direct access.
"""
from deepal6.data.tabular import CreditNet
from deepal6.data.image import _build_resnet18 as build_resnet18

__all__ = ["CreditNet", "build_resnet18"]

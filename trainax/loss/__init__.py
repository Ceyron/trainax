"""
A loss is a time-level comparison between two discrete states.
"""


from .base_loss import BaseLoss
from .l2_loss import L2Loss

__all__ = ["BaseLoss", "L2Loss"]

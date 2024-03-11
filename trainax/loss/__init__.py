"""
A loss is a time-level comparison between two discrete states.
"""


from ._base_loss import BaseLoss
from ._l2_loss import L2Loss

__all__ = ["BaseLoss", "L2Loss"]

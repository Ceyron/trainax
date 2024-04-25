"""
A loss is a time-level comparison between two discrete states. It can also be a
reduction from one discrete state to a scalar value.

All losses shall operate on single batch only. Hence, their expected signature
is `loss_fn(y_true: Array, y_pred: Array) -> float` where `Array` is a JAX numpy
array of shape `(num_channels, ..., num_points)`. The ellipsis indicates an
arbitrary number of spatial axes (potentially of different sizes).

!!! Important: If you want to compute the loss on a batch, i.e., an array with
an additional leading batch axis, use `jax.vmap` on the `loss_fn`. Then, you can
aggregate/reduce the batch axis, for example, with a mean via `jax.numpy.mean`.
"""


from ._base_loss import BaseLoss
from ._l2_loss import L2Loss

__all__ = ["BaseLoss", "L2Loss"]

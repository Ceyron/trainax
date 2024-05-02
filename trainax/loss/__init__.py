"""
A loss is a time-level comparison between two discrete states. It can also be a
reduction from one discrete state to a scalar value.

All losses need to implement a function to work on a single batch. Hence, their
expected signature is `loss_fn.single_batch(y_true: Array, y_pred: Array) ->
float` where `Array` is a JAX numpy array of shape `(num_channels, ...,)`. The
ellipsis indicates an arbitrary number of spatial axes (potentially of different
sizes). Having no spatial axes is also an option.

!!! Important: If you want to compute the loss on a batch, i.e., an array with
an additional leading batch axis, use `jax.vmap` on the `loss_fn.single_batch`.
Then, you can aggregate/reduce the batch axis, for example, with a mean via
`jax.numpy.mean`. Alternatively, use `loss_fn.multi_batch` or simply the call
method, i.e., `loss_fn(...)`.
"""


from ._base_loss import BaseLoss
from ._mae_loss import MAELoss, Normalized_MAELoss
from ._mse_loss import MSELoss, Normalized_MSELoss

__all__ = ["BaseLoss", "MSELoss", "Normalized_MSELoss", "MAELoss", "Normalized_MAELoss"]

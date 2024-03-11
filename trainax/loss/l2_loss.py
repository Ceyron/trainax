import jax.numpy as jnp
from jaxtyping import Array

from .base_time_level_loss import TimeLevelLoss


class L2Loss(TimeLevelLoss):
    """
    Simple Mean Squared Error loss.
    """

    def __call__(
        self,
        prediction: Array,
        target: Array = None,
    ) -> float:
        if target is None:
            diff = prediction
        else:
            diff = prediction - target
        return jnp.mean(jnp.square(diff))

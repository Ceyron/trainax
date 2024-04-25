import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable, Optional

from ._base_loss import BaseLoss


class MSELoss(BaseLoss):
    def __init__(
        self,
        *,
        batch_reduction: Callable = jnp.mean,
    ):
        """
        Simple Mean Squared Error loss.
        """

        super().__init__(batch_reduction=batch_reduction)
    
    def single_batch(
        self,
        prediction: Float[Array, "num_channels ..."],
        target: Optional[Float[Array, "num_channels ..."]] = None,
    ) -> float:
        if target is None:
            diff = prediction
        else:
            diff = prediction - target
        return jnp.mean(jnp.square(diff))
from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

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


class Normalized_MSELoss(MSELoss):
    def __init__(
        self,
        *,
        batch_reduction: Callable = jnp.mean,
    ):
        """
        Simple Mean Squared Error loss normalized on the target.
        """

        super().__init__(batch_reduction=batch_reduction)

    def single_batch(
        self,
        prediction: Float[Array, "num_channels ..."],
        target: Float[Array, "num_channels ..."],
    ) -> float:
        if target is None:
            raise ValueError("Target must be provided for Normalized MSE Loss")

        diff_mse = super().single_batch(prediction, target)
        target_mse = super().single_batch(target)

        return diff_mse / target_mse

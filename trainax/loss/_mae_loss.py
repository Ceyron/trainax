from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ._base_loss import BaseLoss


class MAELoss(BaseLoss):
    def __init__(
        self,
        *,
        batch_reduction: Callable = jnp.mean,
    ):
        """
        Simple Mean Absolute Error loss.
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
        return jnp.mean(jnp.abs(diff))


class Normalized_MAELoss(MAELoss):
    def __init__(
        self,
        *,
        batch_reduction: Callable = jnp.mean,
    ):
        """
        Simple Mean Absolute Error loss normalized on the target.
        """

        super().__init__(batch_reduction=batch_reduction)

    def single_batch(
        self,
        prediction: Float[Array, "num_channels ..."],
        target: Float[Array, "num_channels ..."],
    ) -> float:
        if target is None:
            raise ValueError("Target must be provided for Normalized MAE Loss")

        diff_mae = super().single_batch(prediction, target)
        target_mae = super().single_batch(target)

        return diff_mae / target_mae

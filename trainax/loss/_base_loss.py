import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Float, Array
from typing import Callable, Optional


class BaseLoss(eqx.Module, ABC):
    batch_reduction: Callable

    def __init__(
        self,
        *,
        batch_reduction: Callable = jnp.mean
    ):
        self.batch_reduction = batch_reduction

    @abstractmethod
    def single_batch(
        self,
        prediction: Float[Array, "num_channels ..."],
        target: Optional[Float[Array, "num_channels ..."]] = None,
    ) -> float:
        pass

    def multi_batch(
        self,
        prediction: Float[Array, "num_batches num_channels ..."],
        target: Optional[Float[Array, "num_batches num_channels ..."]] = None,
    ) -> float:
        if target is None:
            return self.batch_reduction(jax.vmap(
                self.single_batch,
                in_axes=(0, None),
            )(prediction, target))
        else:
            return self.batch_reduction(jax.vmap(
                self.single_batch,
                in_axes=(0, 0),
            )(prediction, target))

    def __call__(
        self,
        prediction: Float[Array, "num_batches num_channels ..."],
        target: Optional[Float[Array, "num_batches num_channels ..."]] = None,
    ) -> float:
        return self.multi_batch(prediction, target)

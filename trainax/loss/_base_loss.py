from abc import ABC, abstractmethod
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class BaseLoss(eqx.Module, ABC):
    batch_reduction: Callable

    def __init__(self, *, batch_reduction: Callable = jnp.mean):
        """Base class for loss functions."""
        self.batch_reduction = batch_reduction

    @abstractmethod
    def single_batch(
        self,
        prediction: Float[Array, "num_channels ..."],
        target: Optional[Float[Array, "num_channels ..."]] = None,
    ) -> float:
        """
        Evaluate the loss for a single sample.

        Inputs must be PyTrees of identical structure with array leafs having at
        least a channel/feature axis, and optionally one or more subsequent axes
        (e.g., spatial axes). There should be **no batch axis**.

        !!! info

            To operate on a batch of inputs, either use `multi_batch` or use
            `jax.vmap` on this method.

        **Arguments:**

        - `prediction`: The predicted values.
        - `target`: The target values.

        **Returns:**

        - The loss value.
        """
        pass

    def multi_batch(
        self,
        prediction: Float[Array, "num_batches num_channels ..."],
        target: Optional[Float[Array, "num_batches num_channels ..."]] = None,
    ) -> float:
        """
        Evaluate the loss for a batch of samples.

        Inputs must be PyTrees of identical structure with array leafs having a
        leading batch axis, a subsequent channel/feature axis, and optionally one
        or more subsequent axes (e.g., spatial axes).

        Uses the batch aggregator function specified during initialization.

        **Arguments:**

        - `prediction`: The predicted values.
        - `target`: The target values.

        **Returns:**

        - The loss value.
        """
        if target is None:
            return self.batch_reduction(
                jax.vmap(
                    self.single_batch,
                    in_axes=(0, None),
                )(prediction, target)
            )
        else:
            return self.batch_reduction(
                jax.vmap(
                    self.single_batch,
                    in_axes=(0, 0),
                )(prediction, target)
            )

    def __call__(
        self,
        prediction: Float[Array, "num_batches num_channels ..."],
        target: Optional[Float[Array, "num_batches num_channels ..."]] = None,
    ) -> float:
        """
        Evaluate the loss for a batch of samples.

        Inputs must be PyTrees of identical structure with array leafs having a
        leading batch axis, a subsequent channel/feature axis, and optionally one
        or more subsequent axes (e.g., spatial axes).

        Uses the batch aggregator function specified during initialization.

        **Arguments:**

        - `prediction`: The predicted values.
        - `target`: The target values.

        **Returns:**

        - The loss value.
        """
        return self.multi_batch(prediction, target)

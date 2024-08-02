import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._base import BaseCallback


class WeightNorm(BaseCallback):
    squared: bool = False

    def __init__(self, every: int, squared: bool = False, name: str = "weight_norm"):
        """
        Callback to save the weight norm `every` update steps.

        **Arguments:**

        - `every`: The frequency of the callback.
        - `squared`: Whether to return the squared weight norm.
        - `name`: The name of the callback
        """
        self.squared = squared
        super().__init__(every, name)

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> eqx.Module:
        weights = jtu.tree_leaves(eqx.filter(stepper, eqx.is_array))
        norms_squared = [jnp.sum(w**2) for w in weights]
        norm_squared = sum(norms_squared)

        if self.squared:
            return norm_squared
        else:
            return jnp.sqrt(norm_squared)

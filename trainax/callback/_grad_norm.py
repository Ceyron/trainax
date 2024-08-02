import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from ..configuration import BaseConfiguration
from ._base import BaseCallback


class GradNorm(BaseCallback):
    loss_configuration: BaseConfiguration
    squared: bool

    ref_stepper: eqx.Module
    residuum_fn: eqx.Module

    def __init__(
        self,
        every: int,
        loss_configuration: BaseConfiguration,
        *,
        squared: bool = False,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
        name: str,
    ):
        """
        Callback to save the gradient norm associated with `loss_configuration`
        `every` update steps.

        **Arguments:**

        - `every`: The frequency of the callback.
        - `loss_configuration`: The loss configuration to compute the gradient
            norm. If the gradient norm associated with the training loss is
            desired, the corresponding loss configuration has to be re-supplied.
        - `squared`: Whether to return the squared gradient norm.
        - `ref_stepper`: A reference stepper that is used to compute the residuum.
            Supply this if the loss configuration requires a reference stepper.
        - `residuum_fn`: A residuum function that computes the discrete residuum
            between two consecutive states. Supply this if the loss configuration
            requires a residuum function.
        - `name`: The name of the callback.
        """
        self.loss_configuration = loss_configuration
        self.squared = squared
        self.ref_stepper = ref_stepper
        self.residuum_fn = residuum_fn
        super().__init__(every, name)

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> eqx.Module:
        """Compute the gradient norm."""
        grad = eqx.filter_grad(self.loss_configuration)(
            stepper,
            data,
            ref_stepper=self.ref_stepper,
            residuum_fn=self.residuum_fn,
        )
        grad_weights = jtu.tree_leaves(eqx.filter(grad, eqx.is_array))
        grad_norms_squared = [jnp.sum(g**2) for g in grad_weights]
        grad_norm_squared = sum(grad_norms_squared)
        if self.squared:
            return grad_norm_squared
        else:
            return jnp.sqrt(grad_norm_squared)

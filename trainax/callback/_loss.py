from typing import Union

import equinox as eqx
from jaxtyping import PyTree

from ..configuration import BaseConfiguration
from ._base import BaseCallback


class Loss(BaseCallback):
    loss_configuration: BaseConfiguration
    with_grad: bool

    ref_stepper: eqx.Module
    residuum_fn: eqx.Module

    def __init__(
        self,
        every: int,
        loss_configuration: BaseConfiguration,
        *,
        with_grad: bool = False,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
        name: str,
    ):
        """
        Callback to save the loss associated with `loss_configuration` `every`
        update steps.

        Use this to measure a stepper performance on a difference configuration
        than the training loss.

        **Arguments:**

        - `every`: The frequency of the callback.
        - `loss_configuration`: The loss configuration to compute the loss.
        - `with_grad`: Whether to also return the associated gradient. If only
            the gradient norm is desired, set this to `False` and consider using
            [`trainax.callback.GradNorm`]().
        - `ref_stepper`: A reference stepper that is used to compute the residuum.
            Supply this if the loss configuration requires a reference stepper.
        - `residuum_fn`: A residuum function that computes the discrete residuum
            between two consecutive states. Supply this if the loss configuration
            requires a residuum function.
        - `name`: The name of the callback.
        """
        self.loss_configuration = loss_configuration
        self.with_grad = with_grad
        self.ref_stepper = ref_stepper
        self.residuum_fn = residuum_fn
        super().__init__(every, name)

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Union[eqx.Module, tuple[eqx.Module, eqx.Module]]:
        """
        Compute the loss and optionally the associated gradient.
        """
        if self.with_grad:
            loss, grad = eqx.filter_value_and_grad(self.loss_configuration)(
                stepper,
                data,
                ref_stepper=self.ref_stepper,
                residuum_fn=self.residuum_fn,
            )
            return loss, grad
        else:
            loss = self.loss_configuration(
                stepper,
                data,
                ref_stepper=self.ref_stepper,
                residuum_fn=self.residuum_fn,
            )
            return loss

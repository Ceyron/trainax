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
        self.every = every
        self.loss_configuration = loss_configuration
        self.with_grad = with_grad
        self.ref_stepper = ref_stepper
        self.residuum_fn = residuum_fn
        self.name = name

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> eqx.Module:
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

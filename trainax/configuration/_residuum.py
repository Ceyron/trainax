from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, MSELoss
from ._base_configuration import BaseConfiguration


class Residuum(BaseConfiguration):
    num_rollout_steps: int
    time_level_loss: BaseLoss
    cut_bptt: bool
    cut_bptt_every: int
    cut_prev: bool
    cut_next: bool
    time_level_weights: Float[Array, "num_rollout_steps"]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        *,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_prev: bool = False,
        cut_next: bool = False,
        time_level_weights: Optional[Float[Array, "num_rollout_steps"]] = None,
    ):
        self.num_rollout_steps = num_rollout_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        self.cut_prev = cut_prev
        self.cut_next = cut_next
        if time_level_weights is None:
            self.time_level_weights = jnp.ones(self.num_rollout_steps)
        else:
            self.time_level_weights = time_level_weights

    def __call__(
        self,
        stepper: eqx.Module,
        data: PyTree[Float[Array, "batch num_snapshots ..."]],
        *,
        ref_stepper: eqx.Module = None,  # unused
        residuum_fn: eqx.Module = None,  # unused
    ) -> float:
        # Data is supposed to contain the initial condition, trj is not used
        ic, _ = extract_ic_and_trj(data)

        pred_prev = ic
        loss = 0.0

        for t in range(self.num_rollout_steps):
            pred_next = jax.vmap(stepper)(pred_prev)
            if self.cut_prev:
                pred_prev_mod = jax.lax.stop_gradient(pred_prev)
            else:
                pred_prev_mod = pred_prev
            if self.cut_next:
                pred_next_mod = jax.lax.stop_gradient(pred_next)
            else:
                pred_next_mod = pred_next

            loss += self.time_level_weights[t] * self.time_level_loss(
                residuum_fn(pred_next_mod, pred_prev_mod)
            )

            if self.cut_bptt and (t + 1) % self.cut_bptt_every == 0:
                pred_prev = jax.lax.stop_gradient(pred_next)
            else:
                pred_prev = pred_next

        return loss

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, L2Loss
from ._base_configuration import BaseConfiguration


class DivertedChainBranchOne(BaseConfiguration):
    num_rollout_steps: int
    time_level_loss: BaseLoss
    cut_bptt: bool
    cut_bptt_every: int
    cut_div_chain: bool
    time_level_weights: Float[Array, "num_rollout_steps"]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        *,
        time_level_loss: BaseLoss = L2Loss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: Optional[Float[Array, "num_rollout_steps"]] = None,
    ):
        self.num_rollout_steps = num_rollout_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        self.cut_div_chain = cut_div_chain
        if time_level_weights is None:
            self.time_level_weights = jnp.ones(self.num_rollout_steps)
        else:
            self.time_level_weights = time_level_weights

    def __call__(
        self,
        stepper: eqx.Module,
        data: PyTree[Float[Array, "batch num_snapshots ..."]],
        *,
        ref_stepper: eqx.Module,
        residuum_fn: eqx.Module = None,  # unused
    ) -> float:
        # Data is supposed to contain the initial condition, trj is not used
        ic, _ = extract_ic_and_trj(data)

        pred = ic
        loss = 0.0

        for t in range(self.num_rollout_steps):
            ref = jax.vmap(ref_stepper)(pred)
            if self.cut_div_chain:
                ref = jax.lax.stop_gradient(ref)
            pred = jax.vmap(stepper)(pred)
            loss += self.time_level_weights[t] * self.time_level_loss(pred, ref)

            if self.cut_bptt:
                if (t + 1) % self.cut_bptt_every == 0:
                    pred = jax.lax.stop_gradient(pred)

        return loss
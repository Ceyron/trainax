import jax

from typing import Optional, Union
from abc import ABC, abstractmethod
import equinox as eqx

from jaxtyping import Float, Array, PyTree

from .base_loss_configuration import LossConfiguration
from ..time_level_loss import TimeLevelLoss, L2Loss

from ..utils import extract_ic_and_trj

class DivertedChainBranchOne(LossConfiguration):
    num_rollout_steps: int
    time_level_loss: TimeLevelLoss
    cut_bptt: bool
    cut_bptt_every: int
    cut_div_chain: bool
    time_level_weights: list[float]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        *,
        time_level_loss: TimeLevelLoss = L2Loss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: Optional[list[float]] = None,
    ):
        self.num_rollout_steps = num_rollout_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        self.cut_div_chain = cut_div_chain
        if time_level_weights is None:
            self.time_level_weights = [1.0,] * self.num_rollout_steps
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
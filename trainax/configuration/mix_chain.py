from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree

from ..loss import L2Loss, TimeLevelLoss
from ..utils import extract_ic_and_trj
from .base_loss_configuration import LossConfiguration


class MixChainPostPhysics(LossConfiguration):
    num_rollout_steps: int
    time_level_loss: TimeLevelLoss
    num_post_physics_steps: int
    cut_bptt: bool
    cut_bptt_every: int
    time_level_weights: list[float]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        *,
        time_level_loss: TimeLevelLoss = L2Loss(),
        num_post_physics_steps: int = 1,
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: Optional[list[float]] = None,
    ):
        self.num_rollout_steps = num_rollout_steps
        self.num_post_physics_steps = num_post_physics_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        if time_level_weights is None:
            self.time_level_weights = [
                1.0,
            ] * (self.num_rollout_steps + self.num_post_physics_steps)
        else:
            self.time_level_weights = time_level_weights

    def __call__(
        self,
        stepper: eqx.Module,
        data: PyTree[Float[Array, "batch num_snapshots ..."]],
        *,
        ref_stepper: eqx.Module,  # unused
        residuum_fn: eqx.Module = None,  # unused
    ) -> float:
        # Data is supposed to contain both the initial condition and the target
        ic, trj = extract_ic_and_trj(data)

        # The trj needs to have at least as many snapshots as the number of
        # rollout steps and post physics steps
        if trj.shape[1] < (self.num_rollout_steps + self.num_post_physics_steps):
            raise ValueError(
                "The number of snapshots in the trajectory is less than the "
                "number of rollout steps and post physics steps"
            )

        pred = ic
        loss = 0.0

        # Supervised part
        for t in range(self.num_rollout_steps):
            pred = jax.vmap(stepper)(pred)
            ref = trj[:, t]
            loss += self.time_level_weights[t] * self.time_level_loss(pred, ref)
            if self.cut_bptt:
                if (t + 1) % self.cut_bptt_every == 0:
                    pred = jax.lax.stop_gradient(pred)

        # Post physics part
        for t in range(
            self.num_rollout_steps, self.num_rollout_steps + self.num_post_physics_steps
        ):
            pred = jax.vmap(ref_stepper)(pred)
            ref = trj[:, t]
            loss += self.time_level_weights[t] * self.time_level_loss(pred, ref)

        return loss
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, MSELoss
from ._base_configuration import BaseConfiguration


class DivertedChain(BaseConfiguration):
    num_rollout_steps: int
    num_branch_steps: int
    time_level_loss: BaseLoss
    cut_bptt: bool
    cut_bptt_every: int
    cut_div_chain: bool
    time_level_weights: Float[Array, "num_rollout_steps"]
    branch_level_weights: Float[Array, "num_branch_steps"]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        num_branch_steps: int = 1,
        *,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: Optional[Float[Array, "num_rollout_steps"]] = None,
        branch_level_weights: Optional[Float[Array, "num_branch_steps"]] = None,
    ):
        """
        Most general configuration for diverted chain training. When setting
        `num_branch_steps` to 1, this configuration is equivalent to
        `DivertedChainBranchOne`. (which is probably a better starting point for
        new users.)

        The implementation is rather inefficient!

              L(θ) = 𝔼ᵤ  [ ∑ₜ₌₁ᵀ⁻ᴮ ∑_b₌₁ᴮ wₜ w_b l(  f_θᵗ⁺ᵇ(u),  𝒫ᵇ(f_θᵗ(u)) ) ]
        """
        if num_branch_steps > num_rollout_steps:
            raise ValueError(
                "num_branch_steps must be less than or equal to num_rollout_steps"
            )

        self.num_rollout_steps = num_rollout_steps
        self.num_branch_steps = num_branch_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        self.cut_div_chain = cut_div_chain
        if time_level_weights is None:
            self.time_level_weights = jnp.ones(self.num_rollout_steps)
        else:
            self.time_level_weights = time_level_weights
        if branch_level_weights is None:
            self.branch_level_weights = jnp.ones(self.num_branch_steps)
        else:
            self.branch_level_weights = branch_level_weights

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

        loss = 0.0

        main_chain_pred = ic

        for t in range(self.num_rollout_steps - self.num_branch_steps + 1):
            loss_this_branch = 0.0

            branch_pred = main_chain_pred
            if self.cut_div_chain:
                branch_ref = jax.lax.stop_gradient(main_chain_pred)
            else:
                branch_ref = main_chain_pred
            for b in range(self.num_branch_steps):
                branch_pred = jax.vmap(stepper)(branch_pred)
                branch_ref = jax.vmap(ref_stepper)(branch_ref)
                loss_this_branch += self.branch_level_weights[b] * self.time_level_loss(
                    branch_pred, branch_ref
                )

                if self.cut_bptt:
                    if ((t + b) + 1) % self.cut_bptt_every == 0:
                        branch_pred = jax.lax.stop_gradient(branch_pred)

            loss += self.time_level_weights[t] * loss_this_branch

            main_chain_pred = jax.vmap(stepper)(main_chain_pred)

            if self.cut_bptt:
                if (t + 1) % self.cut_bptt_every == 0:
                    main_chain_pred = jax.lax.stop_gradient(main_chain_pred)

        return loss

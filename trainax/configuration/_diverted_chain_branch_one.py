from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, MSELoss
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
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: Optional[Float[Array, "num_rollout_steps"]] = None,
    ):
        """
        Diverted chain (rollout) configuration with branch length fixed to one.

        Essentially, this amounts to a one-step difference to a reference
        (create on the fly by the differentiable `ref_stepper`). Falls back to
        classical one-step supervised training for `num_rollout_steps=1`
        (default).

        **Arguments:**

        - `num_rollout_steps`: The number of time steps to
            autoregressively roll out the model.Defaults to 1.
        - `time_level_loss`: The loss function to use at
            each time step. Defaults to MSELoss().
        - `cut_bptt`: Whether to cut the backpropagation through time
            (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
            autoregressive network main chain. Defaults to False.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning
            after each step).
        - `cut_div_chain`: Whether to cut the diverted chain, i.e.,
            insert a `jax.lax.stop_gradient` to not have cotangents flow
            over the `ref_stepper`. In this case, the `ref_stepper` does not
            have to be differentiable.
        - `time_level_weights`: An array of length
            `num_rollout_steps` that contains the weights for each time
            step. Defaults to None, which means that all time steps have the
            same weight (=1.0).

        !!! info
            * The `ref_stepper` is called on-the-fly. If its forward (and vjp)
                execution are expensive, this will dominate the computational
                cost of this configuration.
            * The usage of the `ref_stepper` includes the first branch starting
                from the initial condition. Hence, no reference trajectory is
                required.
            * Under reverse-mode automatic differentiation memory usage grows
                linearly with `num_rollout_steps`.
        """
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
        """
        Evaluate the diverted chain (rollout) configuration on the given data.

        The data only has to contain one time level, the initial condition.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            depends on the concrete configuration. In this case, it only
            contains the set of initial states.
        - `ref_stepper`: The reference stepper to use for the
            diverted chain. This is called on-the-fly.
        - `residuum_fn`: For compatibility with other
            configurations; not used.

        **Returns:**

        - The loss value computed by this configuration.
        """
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

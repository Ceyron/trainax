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
        General diverted chain (rollout) configuration.

        Contains the `Supervised` configuration as special case of
        `num_branch_steps=num_rollout_steps` and the `DivertedChainBranchOne`
        configuration as special case of `num_branch_steps=1`.

        **Arguments:**

        - `num_rollout_steps`: The number of time steps to
            autoregressively roll out the model. Defaults to 1.
        - `num_branch_steps`: The number of time steps to branch off the
            main chain. Must be less than or equal to `num_rollout_steps`.
            Defaults to 1.
        - `time_level_loss`: The loss function to use at
            each time step. Defaults to MSELoss().
        - `cut_bptt`: Whether to cut the backpropagation through time
            (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
            autoregressive network main chain. Defaults to False.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning after
            each step).
        - `cut_div_chain`: Whether to cut the diverted chain, i.e.,
            insert a `jax.lax.stop_gradient` to not have cotangents flow over
            the `ref_stepper`. In this case, the `ref_stepper` does not have to
            be differentiable. Defaults to False.
        - `time_level_weights`: An array of length
            `num_rollout_steps` that contains the weights for each time step.
            Defaults to None, which means that all time steps have the same
            weight (=1.0).
        - `branch_level_weights`: An array of length
            `num_branch_steps` that contains the weights for each branch step.
            Defaults to None, which means that all branch steps have the same
            weight (=1.0).

        **Raises:**

        - ValueError: If `num_branch_steps` is greater than
            `num_rollout_steps`.

        !!! info
            * The `ref_stepper` is called on-the-fly. If its forward (and vjp)
                evaluation is expensive, this will dominate the computational
                cost of this configuration.
            * The usage of the `ref_stepper` includes the first branch starting
                from the initial condition. Hence, no reference trajectory is
                required.
            * Under reverse-mode automatic differentiation memory usage grows
                with the product of `num_rollout_steps` and `num_branch_steps`.
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
        """
        Evaluate the general diverted chain (rollout) configuration on the given
        data.

        The data only has to contain one time level, the initial condition.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            depends on the concrete configuration. In this case, it only
            has to contain the set of initial states.
        - `ref_stepper`: The reference stepper to use for the
            diverted chain. This is called on-the-fly.
        - `residuum_fn`: For compatibility with other
            configurations; not used.

        **Returns:**

        - The loss value computed by this configuration.
        """
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

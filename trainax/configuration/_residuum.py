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
        """
        Residuum (rollout) configuration for residua between two consecutive
        time levels.

        If the `ref_stepper` resolves the `residuum_fn` for `u_next`, `Residuum`
        configuration and `DivertedChainBranchOne` configuration can be bounded
        (their losses can be bounded). We believe, however, that both induce
        different optimization trajectories (and different local optima) because
        the residuum-based loss is conditioned worse.

        **Arguments:**

        - `num_rollout_steps`: The number of time steps to
            autoregressively roll out the model. Defaults to 1.
        - `time_level_loss`: The loss function to use at
            each time step. Must operate based on a single input. Defaults to
            MSELoss().
        - `cut_bptt`: Whether to cut the backpropagation through time
            (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
            autoregressive network main chain. Defaults to False.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning after
            each step).
        - `cut_prev`: Whether to cut the previous time level contribution
            to `residuum_fn`. Defaults to False.
        - `cut_next`: Whether to cut the next time level contribution
            to `residuum_fn`. Defaults to False.
        - `time_level_weights`: An array of length `num_rollout_steps` that
            contains the weights for each time step. Defaults to None, which
            means that all time steps have the same weight (=1.0).
        """
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
        residuum_fn: eqx.Module,
    ) -> float:
        """
        Evaluate the residuum (rollout) configuration on the given data.

        The data only has to contain one time level, the initial condition. The
        `residuum_fn` will be used to compute a loss based on two consecutive
        time levels.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            depends on the concrete configuration. In this case, it only
            contains the initial condition.
        - `ref_stepper`: The reference stepper to use for the
            configuration. Must have the signature `ref_stepper(u_prev: PyTree)
            -> u_next: PyTree`. Defaults to None.
        - `residuum_fn`: The residuum function to use for the
            configuration. Must have the signature `residuum_fn(u_next: PyTree,
            u_prev: PyTree) -> residuum: PyTree`.

        **Returns:**

        - The loss of the configuration.
        """
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

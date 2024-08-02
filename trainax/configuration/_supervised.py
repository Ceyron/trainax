from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, MSELoss
from ._base_configuration import BaseConfiguration


class Supervised(BaseConfiguration):
    num_rollout_steps: int
    time_level_loss: BaseLoss
    cut_bptt: bool
    cut_bptt_every: int
    time_level_weights: Float[Array, "num_rollout_steps"]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        *,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: Optional[Float[Array, "num_rollout_steps"]] = None,
    ):
        """
        General supervised (rollout) configuration.

        Falls back to classical one-step supervised training for
        `num_rollout_steps=1` (default).

        **Arguments:**

        - `num_rollout_steps`: The number of time steps to
            autoregressively roll out the model. During calling this
            configuration, it requires a similarly long reference trajectory to
            be available. Defaults to 1.
        - `time_level_loss`: The loss function to use at
            each time step. Defaults to MSELoss().
        - `cut_bptt`: Whether to cut the backpropagation through time
            (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
            autoregressive network main chain. Defaults to False.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning after
            each step).
        - `time_level_weights`: An array of length
            `num_rollout_steps` that contains the weights for each time step.
            Defaults to None, which means that all time steps have the same
            weight (=1.0).


        !!! warning
            Under reverse-mode automatic differentiation memory usage grows
            linearly with `num_rollout_steps`.
        """
        self.num_rollout_steps = num_rollout_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
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
        """
        Evaluate the supervised (rollout) configuration on the given data.

        The data is supposed to have as many time steps as the number of rollout
        steps plus one. No `ref_stepper` or `residuum_fn` is needed.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            should contain the initial condition and the target trajectory.
        - `ref_stepper`: For compatibility with other
            configurations; not used.
        - `residuum_fn`: For compatibility with other
            configurations; not used.

        **Returns:**

        - The loss value computed by this configuration.

        **Raises:**

        - ValueError: If the number of snapshots in the trajectory is less than
            the number of rollout steps plus one.
        """
        # Data is supposed to contain both the initial condition and the target
        ic, trj = extract_ic_and_trj(data)

        # The trj needs to have at least as many snapshots as the number of
        # rollout steps
        if trj.shape[1] < self.num_rollout_steps:
            raise ValueError(
                "The number of snapshots in the trajectory is less than the "
                "number of rollout steps"
            )

        pred = ic
        loss = 0.0

        for t in range(self.num_rollout_steps):
            pred = jax.vmap(stepper)(pred)
            ref = trj[:, t]
            loss += self.time_level_weights[t] * self.time_level_loss(pred, ref)
            if self.cut_bptt:
                if (t + 1) % self.cut_bptt_every == 0:
                    pred = jax.lax.stop_gradient(pred)

        return loss

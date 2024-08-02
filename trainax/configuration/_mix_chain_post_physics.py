from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from .._utils import extract_ic_and_trj
from ..loss import BaseLoss, MSELoss
from ._base_configuration import BaseConfiguration


class MixChainPostPhysics(BaseConfiguration):
    num_rollout_steps: int
    time_level_loss: BaseLoss
    num_post_physics_steps: int
    cut_bptt: bool
    cut_bptt_every: int
    time_level_weights: Float[Array, "num_rollout_steps+num_post_physics_steps"]

    def __init__(
        self,
        num_rollout_steps: int = 1,
        num_post_physics_steps: int = 1,
        *,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: Optional[
            Float[Array, "num_rollout_steps+num_post_physics_steps"]
        ] = None,
    ):
        """
        Mix chain (rollout) configuration with autoregressive physics steps
        after the autoregressive emulator steps in the main chain.

        This is a special case of potentially more complicated combitations of
        neural stepper with reference physics stepper in the main chain.

        **Arguments:**

        - `num_rollout_steps`: The number of time steps to
            autoregressively roll out the model. Defaults to 1.
        - `num_post_physics_steps`: The number of time steps to
            autoregressively roll physics **after** the model in the main chain.
            Defaults to 1. Hence, in the default config, the main chain is model
            -> physics
        - `time_level_loss`: The loss function to use at
            each time step. Defaults to `trainax.loss.MSELoss`.
        - `cut_bptt`: Whether to cut the backpropagation through time
            (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
            autoregressive network main chain. This excludes the post-physics
            steps; those are not cutted. Defaults to False.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning after
            each step).
        - `time_level_weights`: An array of length
            `num_rollout_steps+num_post_physics_steps` that contains the weights
            for each time step. Defaults to None, which means that all time
            steps have the same weight (=1.0).
        """
        self.num_rollout_steps = num_rollout_steps
        self.num_post_physics_steps = num_post_physics_steps
        self.time_level_loss = time_level_loss
        self.cut_bptt = cut_bptt
        self.cut_bptt_every = cut_bptt_every
        if time_level_weights is None:
            self.time_level_weights = jnp.ones(
                self.num_rollout_steps + self.num_post_physics_steps
            )
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
        Evaluate the mix chain (rollout) configuration on the given data.

        The data only has to contain as many time levels as the sum of the
        number of rollout steps and post physics steps plus one.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. Has to
            contain the initial condition and the target trajectory.
        - `ref_stepper`: The reference stepper to use for the
            configuration. Must have the signature `ref_stepper(u_prev: PyTree)
            -> u_next: PyTree`.
        - `residuum_fn`: For compatibility with other
            configurations; not used here.

        **Returns:**

        - The loss value computed by this configuration.

        **Raises:**

        - ValueError: If the number of snapshots in the trajectory is less than
            the number of rollout steps and post physics steps plus one.
        """
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

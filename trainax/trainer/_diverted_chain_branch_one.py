from typing import Optional

import equinox as eqx
import optax
from jaxtyping import Array, Float

from .._general_trainer import GeneralTrainer
from .._mixer import TrajectorySubStacker
from ..callback import BaseCallback
from ..configuration import DivertedChainBranchOne
from ..loss import BaseLoss, MSELoss


class DivertedChainBranchOneTrainer(GeneralTrainer):
    def __init__(
        self,
        data_trajectories,
        *,
        ref_stepper: eqx.Module,
        residuum_fn: eqx.Module = None,  # for compatibility
        optimizer: optax.GradientTransformation,
        callback_fn: Optional[BaseCallback] = None,
        num_training_steps: int,
        batch_size: int,
        num_rollout_steps: int = 1,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: Optional[
            Float[Array, "num_rollout_steps"]  # noqa F821
        ] = None,
        do_sub_stacking: bool = True,
    ):
        """
        Diverted chain (rollout) configuration with branch length fixed to one.

        Essentially, this amounts to a one-step difference to a reference
        (created on the fly by the differentiable `ref_stepper`). Falls back to
        classical one-step supervised training for `num_rollout_steps=1`
        (default).

        **Arguments:**

        - `data_trajectories`: The batch of trajectories to slice. This must be
            a PyTree of Arrays who have at least two leading axes: a batch-axis
            and a time axis. For example, the zeroth axis can be associated with
            multiple initial conditions or constitutive parameters and the first
            axis represents all temporal snapshots. A PyTree can also just be an
            array. You can provide additional leafs in the PyTree, e.g., for the
            corresponding constitutive parameters etc. Make sure that the
            emulator has the corresponding signature.
        - `ref_stepper`: The reference stepper to use for the diverted chain.
            This is called on-the-fly.
        - `residuum_fn`: For compatibility with other configurations; not used.
        - `optimizer`: The optimizer to use for training. For example, this can
            be `optax.adam(LEARNING_RATE)`. Also use this to supply an optimizer
            with learning rate decay, for example
            `optax.adam(optax.exponential_decay(...))`. If your learning rate
            decay is designed for a certain number of update steps, make sure
            that it aligns with `num_training_steps`.
        - `callback_fn`: A callback to use during training. Defaults to None.
        - `num_training_steps`: The number of training steps to perform.
        - `batch_size`: The batch size to use for training. Batches are
            randomly sampled across both multiple trajectories, but also over
            different windows within one trajectory.
        - `num_rollout_steps: The number of time steps to autoregressively
            roll out the model.
        - `time_level_loss`: The loss function to use at each time step.
        - `cut_bptt`: Whether to cut the backpropagation through time (BPTT),
            i.e., insert a `jax.lax.stop_gradient` into the autoregressive
            network main chain.
        - `cut_bptt_every`: The frequency at which to cut the BPTT. Only
            relevant if `cut_bptt` is True. Defaults to 1 (meaning after each
            step).
        - `cut_div_chain`: Whether to cut the diverted chain, i.e.,
            insert a `jax.lax.stop_gradient` to not have cotangents flow over
            the `ref_stepper`. In this case, the `ref_stepper` does not have to
            be differentiable.
        - `time_level_weights`: An array of length `num_rollout_steps` that
            contains the weights for each time step. Defaults to None, which
            means that all time steps have the same weight (=1.0). (keyword-only
            argument)


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
        trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1,  # +1 for the IC
            do_sub_stacking=do_sub_stacking,
            only_store_ic=True,  # Not needed because we use the ref_stepper
        )
        loss_configuration = DivertedChainBranchOne(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
            cut_div_chain=cut_div_chain,
            time_level_weights=time_level_weights,
        )
        super().__init__(
            trajectory_sub_stacker,
            loss_configuration,
            ref_stepper=ref_stepper,
            residuum_fn=residuum_fn,
            optimizer=optimizer,
            num_minibatches=num_training_steps,
            batch_size=batch_size,
            callback_fn=callback_fn,
        )

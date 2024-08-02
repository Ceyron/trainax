from typing import Optional

import equinox as eqx
import optax
from jaxtyping import Array, Float, PyTree

from .._general_trainer import GeneralTrainer
from .._mixer import TrajectorySubStacker
from ..callback import BaseCallback
from ..configuration import Residuum
from ..loss import BaseLoss, MSELoss


class ResiduumTrainer(GeneralTrainer):
    def __init__(
        self,
        data_trajectories: PyTree[Float[Array, "num_samples trj_len ..."]],
        *,
        ref_stepper: eqx.Module = None,  # for compatibility
        residuum_fn: eqx.Module,
        optimizer: optax.GradientTransformation,
        callback_fn: Optional[BaseCallback] = None,
        num_training_steps: int,
        batch_size: int,
        num_rollout_steps: int = 1,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_prev: bool = False,
        cut_next: bool = False,
        time_level_weights: Optional[
            Float[Array, "num_rollout_steps"]  # noqa F821
        ] = None,
        do_sub_stacking: bool = True,
    ):
        """
        Residuum (rollout) training for an autoregressive neural emulator on a
        collection of trajectories.

        If the `ref_stepper` resolves the `residuum_fn` for `u_next`, `Residuum`
        configuration and `DivertedChainBranchOne` configuration can be bounded
        (their losses can be bounded). We believe, however, that both induce
        different optimization trajectories (and different local optima) because
        the residuum-based loss is conditioned worse.

        **Arguments:**

        - `data_trajectories`: The batch of trajectories to slice. This must be
            a PyTree of Arrays who have at least two leading axes: a batch-axis
            and a time axis. For example, the zeroth axis can be associated with
            multiple initial conditions or constitutive parameters and the first
            axis represents all temporal snapshots. A PyTree can also just be an
            array. You can provide additional leafs in the PyTree, e.g., for the
            corresponding constitutive parameters etc. Make sure that the
            emulator has the corresponding signature.
        - `ref_stepper`: For compatibility with other configurations; not used.
        - `residuum_fn`: The residuum function to use for the configuration.
            Must have the signature `residuum_fn(u_next: PyTree, u_prev: PyTree)
            -> residuum: PyTree`.
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
        - `num_rollout_steps`: The number of time steps to autoregressively roll
            out the model during training.
        - `time_level_loss`: The loss function to use at each time step.
        - `cut_bptt`: Whether to cut the backpropagation through time (BPTT),
            i.e., insert a `jax.lax.stop_gradient` into the autoregressive
            network main chain.
        - `cut_bptt_every`: The frequency at which to cut the BPTT.
            Only relevant if `cut_bptt` is True. Defaults to 1 (meaning after
            each step).
        - `cut_prev`: Whether to cut the previous time level contribution
            to `residuum_fn`.
        - `cut_next`: Whether to cut the next time level contribution
            to `residuum_fn`.
        - `time_level_weights: An array of length `num_rollout_steps` that
            contains the weights for each time step. Defaults to None, which
            means that all time steps have the same weight (=1.0).

        !!! info
            * Under reverse-mode automatic differentiation memory usage grows
                linearly with `num_rollout_steps`.
        """
        trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1,  # +1 for the IC
            do_sub_stacking=do_sub_stacking,
            only_store_ic=False,
        )
        loss_configuration = Residuum(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
            cut_prev=cut_prev,
            cut_next=cut_next,
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

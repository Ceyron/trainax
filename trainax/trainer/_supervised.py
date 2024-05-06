from typing import Optional

import equinox as eqx
import optax
from jaxtyping import Array, Float, PyTree

from .._general_trainer import GeneralTrainer
from .._mixer import TrajectorySubStacker
from ..callback import BaseCallback
from ..configuration import Supervised
from ..loss import BaseLoss, MSELoss


class SupervisedTrainer(GeneralTrainer):
    def __init__(
        self,
        data_trajectories: PyTree[Float[Array, "num_samples trj_len ..."]],
        *,
        ref_stepper: eqx.Module = None,  # for compatibility
        residuum_fn: eqx.Module = None,  # for compatibility
        optimizer: optax.GradientTransformation,
        callback_fn: Optional[BaseCallback] = None,
        num_training_steps: int,
        batch_size: int,
        num_rollout_steps: int = 1,
        time_level_loss: BaseLoss = MSELoss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: Optional[
            Float[Array, "num_rollout_steps"]  # noqa F821
        ] = None,
        do_sub_stacking: bool = True,
    ):
        """
        Supervised (rollout) trainig for an autoregressive neural emulator on a
        collection of trajectories.

        Classical one-step supervised learning is achieved with
        `num_rollout_steps=1`.

        Ensure that the `data_trajectories` trajectory length is at least as
        long as `num_rollout_steps + 1` (the additional step is for the initial
        condition).

        Args:
            data_trajectories (PyTree[Float[Array, "num_samples trj_len ..."]]):
                The batch of trajectories to slice. This must be a PyTree of
                Arrays who have at least two leading axes: a batch-axis and a
                time axis. For example, the zeroth axis can be associated with
                multiple initial conditions or constitutive parameters and the
                first axis represents all temporal snapshots. A PyTree can also
                just be an array. You can provide additional leafs in the
                PyTree, e.g., for the corresponding constitutive parameters etc.
                Make sure that the emulator has the corresponding signature.
            ref_stepper (eqx.Module): For compatibility with other
                configurations; not used. (keyword-only argument)
            residuum_fn (eqx.Module): For compatibility with other
                configurations; not used. (keyword-only argument)
            optimizer (optax.GradientTransformation): The optimizer to use for
                training. For example, this can be `optax.adam(LEARNING_RATE)`.
                Also use this to supply an optimizer with learning rate decay,
                for example `optax.adam(optax.exponential_decay(...))`. If your
                learning rate decay is designed for a certain number of update
                steps, make sure that it aligns with `num_training_steps`.
                (keyword-only argument)
            callback_fn (BaseCallback, optional): A callback to use during
                training. Defaults to None. (keyword-only argument)
            num_training_steps (int): The number of training steps to perform.
                (keyword-only argument)
            batch_size (int): The batch size to use for training. Batches are
                randomly sampled across both multiple trajectories, but also over
                different windows within one trajectory. (keyword-only)
            num_rollout_steps (int): The number of time steps to
                autoregressively roll out the model. Defaults to 1 (which is one-step
                supervised training). (keyword-only argument)
            time_level_loss (BaseLoss): The loss function to use at
                each time step. Defaults to MSELoss(). (keyword-only argument)
            cut_bptt (bool): Whether to cut the backpropagation through time
                (BPTT), i.e., insert a `jax.lax.stop_gradient` into the
                autoregressive network main chain. Defaults to False.
                (keyword-only argument)
            cut_bptt_every (int): The frequency at which to cut the BPTT.
                Only relevant if `cut_bptt` is True. Defaults to 1 (meaning
                after each step). (keyword-only argument)
            time_level_weights (array[float], optional): An array of length
                `num_rollout_steps` that contains the weights for each time
                step. Defaults to None, which means that all time steps have the
                same weight (=1.0). (keyword-only argument)
            do_sub_stacking (bool): Whether to use sub-stacking. If `False`, then
                the given reference trajectory will not be sliced into windows
                of length `num_rollout_steps + 1`. Defaults to True.
                (keyword-only argument)

        Info:
            * Under reverse-mode automatic differentiation memory usage grows
                linearly with `num_rollout_steps`.
        """
        trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1,  # +1 for the IC
            do_sub_stacking=do_sub_stacking,
            only_store_ic=False,
        )
        loss_configuration = Supervised(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
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

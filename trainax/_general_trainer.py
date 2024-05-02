from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree
from tqdm.autonotebook import tqdm

from ._mixer import PermutationMixer, TrajectorySubStacker
from .callback import BaseCallback
from .configuration import BaseConfiguration


class GeneralTrainer(eqx.Module):
    trajectory_sub_stacker: TrajectorySubStacker
    loss_configuration: BaseConfiguration
    ref_stepper: eqx.Module
    residuum_fn: eqx.Module
    optimizer: optax.GradientTransformation
    num_minibatches: int
    batch_size: int
    callback_fn: BaseCallback

    def __init__(
        self,
        trajectory_sub_stacker: TrajectorySubStacker,
        loss_configuration: BaseConfiguration,
        *,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
        optimizer: optax.GradientTransformation,
        num_minibatches: int,
        batch_size: int,
        callback_fn: Optional[BaseCallback] = None,
    ):
        """
        Abstract training an autoregressive neural emulator on a collection of
        trajectories.

        The length of (sub-)trajectories returned by `trajectory_sub_stacker` must
        match the requires length of reference for the used `loss_configuration`.

        Args:
            trajectory_sub_stacker (TrajectorySubStacker): A callable that takes a
                list of indices and returns a collection of (sub-)trajectories.
            loss_configuration (BaseConfiguration): A configuration that defines the
                loss function to be minimized.
            ref_stepper (eqx.Module, optional): A reference stepper that is used to
                compute the residuum. Supply this if the loss configuration requires
                a reference stepper. Defaults to None.
            residuum_fn (eqx.Module, optional): A residuum function that computes the
                discrete residuum between two consecutive states. Supply this if the
                loss configuration requires a residuum function. Defaults to None.
            optimizer (optax.GradientTransformation): An optimizer that updates the
                parameters of the stepper given the gradient.
            num_minibatches (int): The number of minibatches to train on. This equals
                the total number of update steps performed. The number of epochs is
                determined based on this and the `batch_size`.
            batch_size (int): The size of each batch.
            callback_fn (BaseCallback, optional): A callback function that is called
                at the end of each minibatch. Defaults to None.
        """
        self.trajectory_sub_stacker = trajectory_sub_stacker
        self.loss_configuration = loss_configuration
        self.ref_stepper = ref_stepper
        self.residuum_fn = residuum_fn
        self.optimizer = optimizer
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.callback_fn = callback_fn

    def full_loss(
        self,
        stepper: eqx.Module,
    ) -> float:
        """
        Compute the loss on the entire dataset.
        """
        return self.loss_configuration(
            stepper,
            self.trajectory_sub_stacker.data_sub_trajectories,
            ref_stepper=self.ref_stepper,
            residuum_fn=self.residuum_fn,
        )

    def step_fn(
        self,
        stepper: eqx.Module,
        opt_state: optax.OptState,
        data: PyTree,
    ) -> tuple[eqx.Module, optax.OptState, float]:
        """
        Perform a single update step to the `stepper`'s parameters.

        Args:
            stepper (eqx.Module): The stepper to be updated.
            opt_state (optax.OptState): The optimizer state.
            data (PyTree): The data for the current minibatch.

        Returns:
            tuple[eqx.Module, optax.OptState, float]: The updated stepper, the
                updated optimizer state, and the loss value.
        """
        loss, grad = eqx.filter_value_and_grad(
            lambda m: self.loss_configuration(
                m, data, ref_stepper=self.ref_stepper, residuum_fn=self.residuum_fn
            )
        )(stepper)
        updates, new_opt_state = self.optimizer.update(grad, opt_state, stepper)
        new_stepper = eqx.apply_updates(stepper, updates)
        return new_stepper, new_opt_state, loss

    def __call__(
        self,
        stepper: eqx.Module,
        key: PRNGKeyArray,
        *,
        return_loss_history: bool = True,
        record_loss_every: int = 1,
    ):
        """
        Perform the entire training of an autoregressive neural emulator
        `stepper`.

        This method spawns a `tqdm` progress meter showing the current update
        step and displaying the epoch with its respetive minibatch counter.

        This method's return signature depends on the presence of a callback
        function. If a callback function is provided, this function has at max
        three return values. The first return value is the trained stepper, the
        second return value is the loss history, and the third return value is
        the auxiliary history. The auxiliary history is a list of the return
        values of the callback function at each minibatch. If no callback
        function is provided, this function has at max two return values. The
        first return value is the trained stepper, and the second return value
        is the loss history.

        Args:
            stepper (eqx.Module): The stepper to be trained. key (PRNGKeyArray):
            The random key to be used for shuffling the
                minibatches.
            return_loss_history (bool, optional): Whether to return the loss
                history. Defaults to True.
            record_loss_every (int, optional): Record the loss every
                `record_loss_every` minibatches. Defaults to 1.

        Returns:
            Varying, see above.
        """
        loss_history = []
        if self.callback_fn is not None:
            aux_history = []

        mixer = PermutationMixer(
            num_total_samples=self.trajectory_sub_stacker.num_total_samples,
            num_minibatches=self.num_minibatches,
            batch_size=self.batch_size,
            shuffle_key=key,
        )

        p_meter = tqdm(
            total=self.num_minibatches,
            desc=f"E: {0:05d}, B: {0:05d}",
        )

        update_fn = eqx.filter_jit(self.step_fn)

        trained_stepper = stepper
        opt_state = self.optimizer.init(eqx.filter(trained_stepper, eqx.is_array))

        for update_i in range(self.num_minibatches):
            batch_indices, (expoch_id, batch_id) = mixer(update_i, return_info=True)
            data = self.trajectory_sub_stacker(batch_indices)
            if self.callback_fn is not None:
                aux = self.callback_fn(update_i, trained_stepper, data)
                aux_history.append(aux)
            trained_stepper, opt_state, loss = update_fn(
                trained_stepper, opt_state, data
            )
            if update_i % record_loss_every == 0:
                loss_history.append(loss)
            p_meter.update(1)

            p_meter.set_description(
                f"E: {expoch_id:05d}, B: {batch_id:05d}",
            )

        p_meter.close()

        loss_history = jnp.array(loss_history)

        if self.callback_fn is not None:
            if return_loss_history:
                return trained_stepper, loss_history, aux_history
            else:
                return trained_stepper, aux_history
        else:
            if return_loss_history:
                return trained_stepper, loss_history
            else:
                return trained_stepper

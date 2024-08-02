from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
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
        Abstract training for an autoregressive neural emulator on a collection
        of trajectories.

        !!! info
            The length of (sub-)trajectories returned by
            `trajectory_sub_stacker` must match the required length of reference
            for the used `loss_configuration`.

        **Arguments:**

        - `trajectory_sub_stacker`: A callable that takes a
            list of indices and returns a collection of (sub-)trajectories.
        - `loss_configuration`: A configuration that defines the
            loss function to be minimized.
        - `ref_stepper`: A reference stepper that is used to
            compute the residuum. Supply this if the loss configuration requires
            a reference stepper.
        - `residuum_fn`: A residuum function that computes the
            discrete residuum between two consecutive states. Supply this if the
            loss configuration requires a residuum function. Defaults to None.
        - `optimizer`: An optimizer that updates the
            parameters of the stepper given the gradient.
        - `num_minibatches`: The number of minibatches to train on. This equals
            the total number of update steps performed. The number of epochs is
            automatically determined based on this and the `batch_size`.
        - `batch_size`: The size of each minibatch, i.e., how many samples are
            included within.
        - `callback_fn`: A callback function that is called
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

        !!! warning
            This can lead to out of memory errors if the dataset is too large.

        **Arguments:**

        - `stepper`: The stepper to compute the loss with.

        **Returns:**

        - The loss value.
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
        data: PyTree[Float[Array, "batch_size sub_trj_len ..."]],
    ) -> tuple[eqx.Module, optax.OptState, float]:
        """
        Perform a single update step to the `stepper`'s parameters.

        **Arguments:**

        - `stepper`: The equinox module to be updated.
        - `opt_state`: The current optimizer state.
        - `data`: The data for the current minibatch.

        **Returns:**

        - The updated equinox module
        - The updated optimizer state
        - The loss value
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
        opt_state: Optional[optax.OptState] = None,
        *,
        return_loss_history: bool = True,
        record_loss_every: int = 1,
        spawn_tqdm: bool = True,
    ) -> Union[
        tuple[eqx.Module, Float[Array, "num_minibatches"]],
        eqx.Module,
        tuple[eqx.Module, Float[Array, "num_minibatches"], list],
        tuple[eqx.Module, list],
    ]:
        """
        Perform the entire training of an autoregressive neural emulator given
        in an initial state as `stepper`.

        This method's return signature depends on the presence of a callback
        function. If a callback function is provided, this function has at max
        three return values. The first return value is the trained stepper, the
        second return value is the loss history, and the third return value is
        the auxiliary history. The auxiliary history is a list of the return
        values of the callback function at each minibatch. If no callback
        function is provided, this function has at max two return values. The
        first return value is the trained stepper, and the second return value
        is the loss history. If `return_loss_history` is set to `False`, the
        loss history will not be returned.

        **Arguments:**

        - `stepper`: The equinox Module to be trained.
        - `key`: The random key to be used for shuffling the minibatches.
        - `opt_state`: The initial optimizer state. Defaults to None, meaning
            the optimizer will be reinitialized.
        - `return_loss_history`: Whether to return the loss history.
        - `record_loss_every`: Record the loss every `record_loss_every`
            minibatches. Defaults to 1, i.e., record every minibatch.
        - `spawn_tqdm`: Whether to spawn the tqdm progress meter showing the
            current update step and displaying the epoch with its respetive
            minibatch counter.

        **Returns:**

        - Varying, see above. It will always return the trained stepper as the
            first return value.

        !!! tip
            You can use `equinox.filter_vmap` to train mulitple networks (of the
            same architecture) at the same time. For example, if your GPU is not
            fully utilized yet, this will give you a init-seed statistic
            basically for free.
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

        if spawn_tqdm:
            p_meter = tqdm(
                total=self.num_minibatches,
                desc=f"E: {0:05d}, B: {0:05d}",
            )

        update_fn = eqx.filter_jit(self.step_fn)

        trained_stepper = stepper
        if opt_state is None:
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
            if spawn_tqdm:
                p_meter.update(1)

                p_meter.set_description(
                    f"E: {expoch_id:05d}, B: {batch_id:05d}",
                )

        if spawn_tqdm:
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

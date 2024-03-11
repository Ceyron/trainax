import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray
from tqdm.autonotebook import tqdm

from .loss_configuration import LossConfiguration
from .mixer import PermutationMixer, TrajectorySubStacker


class Trainer(eqx.Module):
    trajectory_sub_stacker: TrajectorySubStacker
    loss_configuration: LossConfiguration
    ref_stepper: eqx.Module
    residuum_fn: eqx.Module
    optimizer: optax.GradientTransformation
    num_minibatches: int
    batch_size: int
    callback_fn: eqx.Module

    def __init__(
        self,
        trajectory_sub_stacker: TrajectorySubStacker,
        loss_configuration: LossConfiguration,
        *,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
        optimizer: optax.GradientTransformation,
        num_minibatches: int,
        batch_size: int,
        callback_fn=None,
    ):
        self.trajectory_sub_stacker = trajectory_sub_stacker
        self.loss_configuration = loss_configuration
        self.ref_stepper = ref_stepper
        self.residuum_fn = residuum_fn
        self.optimizer = optimizer
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.callback_fn = callback_fn

    def step_fn(
        self,
        stepper,
        opt_state,
        data,
    ):
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
    ):
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

import jax
import jax.numpy as jnp
import pytest
from _utils import run

import trainax as tx


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration(num_rollout_steps):
    sup_rollout_config = tx.configuration.Supervised(num_rollout_steps)

    def manual_loss_fn(model, data, *, ref_stepper=None, residuum_fn=None):
        ic = data[:, 0, :]
        ref = data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))

        return loss

    run(sup_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration_cut_bptt(num_rollout_steps):
    """
    Completely cuts the backpropagation through time
    """
    sup_rollout_config = tx.configuration.Supervised(num_rollout_steps, cut_bptt=True)

    def manual_loss_fn(model, data, *, ref_stepper=None, residuum_fn=None):
        ic = data[:, 0, :]
        ref = data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))
            pred = jax.lax.stop_gradient(pred)

        return loss

    run(sup_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5, 6, 7, 8])
def test_supervised_rollout_configuration_sparse_bptt(num_rollout_steps):
    """
    Cuts the bptt every three steps
    """
    sup_rollout_config = tx.configuration.Supervised(
        num_rollout_steps, cut_bptt=True, cut_bptt_every=3
    )

    def manual_loss_fn(model, data, *, ref_stepper=None, residuum_fn=None):
        ic = data[:, 0, :]
        ref = data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))
            if (i + 1) % 3 == 0:
                pred = jax.lax.stop_gradient(pred)

        return loss

    run(sup_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration_time_level_weights(num_rollout_steps):
    """
    Weighs the time level loss differently
    """

    time_level_weights = jax.random.normal(jax.random.PRNGKey(0), (num_rollout_steps,))
    sup_rollout_config = tx.configuration.Supervised(
        num_rollout_steps, time_level_weights=time_level_weights
    )

    def manual_loss_fn(model, data, *, ref_stepper=None, residuum_fn=None):
        ic = data[:, 0, :]
        ref = data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += time_level_weights[i] * jnp.mean(jnp.square(pred - ref[:, i, :]))

        return loss

    run(sup_rollout_config, manual_loss_fn)

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _utils import compare_pytree

import trainax as tx


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration(num_rollout_steps):
    NUM_DOF = 10
    sup_rollout_config = tx.configuration.Supervised(num_rollout_steps)

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]
    shape = (5, num_rollout_steps + 1, NUM_DOF)
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), shape)

    dummy_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(11),
    )

    loss = sup_rollout_config(dummy_mlp, dummy_data)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        ref = dummy_data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(sup_rollout_config)(dummy_mlp, dummy_data)
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration_cut_bptt(num_rollout_steps):
    """
    Completely cuts the backpropagation through time
    """
    NUM_DOF = 10
    sup_rollout_config = tx.configuration.Supervised(num_rollout_steps, cut_bptt=True)

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]
    shape = (5, num_rollout_steps + 1, NUM_DOF)
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), shape)

    dummy_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(11),
    )

    loss = sup_rollout_config(dummy_mlp, dummy_data)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        ref = dummy_data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))
            pred = jax.lax.stop_gradient(pred)

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(sup_rollout_config)(dummy_mlp, dummy_data)
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5, 6, 7, 8])
def test_supervised_rollout_configuration_sparse_bptt(num_rollout_steps):
    """
    Cuts the bptt every three steps
    """
    NUM_DOF = 10
    sup_rollout_config = tx.configuration.Supervised(
        num_rollout_steps, cut_bptt=True, cut_bptt_every=3
    )

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]
    shape = (5, num_rollout_steps + 1, NUM_DOF)
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), shape)

    dummy_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(11),
    )

    loss = sup_rollout_config(dummy_mlp, dummy_data)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        ref = dummy_data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref[:, i, :]))
            if (i + 1) % 3 == 0:
                pred = jax.lax.stop_gradient(pred)

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(sup_rollout_config)(dummy_mlp, dummy_data)
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_supervised_rollout_configuration_time_level_weights(num_rollout_steps):
    """
    Weighs the time level loss differently
    """

    NUM_DOF = 10
    time_level_weights = jax.random.normal(jax.random.PRNGKey(0), (num_rollout_steps,))
    sup_rollout_config = tx.configuration.Supervised(
        num_rollout_steps, time_level_weights=time_level_weights
    )

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]
    shape = (5, num_rollout_steps + 1, NUM_DOF)
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), shape)

    dummy_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(11),
    )

    loss = sup_rollout_config(dummy_mlp, dummy_data)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        ref = dummy_data[:, 1:, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            pred = jax.vmap(model)(pred)
            loss += time_level_weights[i] * jnp.mean(jnp.square(pred - ref[:, i, :]))

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(sup_rollout_config)(dummy_mlp, dummy_data)
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)

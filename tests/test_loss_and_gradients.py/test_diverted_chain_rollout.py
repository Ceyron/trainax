import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _utils import compare_pytree

import trainax as tx


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration(num_rollout_steps):
    NUM_DOF = 10
    div_rollout_config = tx.configuration.DivertedChainBranchOne(num_rollout_steps)

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

    dummy_ref_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(27),
    )

    loss = div_rollout_config(dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(dummy_ref_mlp)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(div_rollout_config)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp
    )
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration_cut_bptt(num_rollout_steps):
    """
    Completely cuts the backpropagation through time
    """
    NUM_DOF = 10
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_bptt=True
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

    dummy_ref_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(27),
    )

    loss = div_rollout_config(dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(dummy_ref_mlp)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

            pred = jax.lax.stop_gradient(pred)

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(div_rollout_config)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp
    )
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration_cut_div_chain(num_rollout_steps):
    """
    Completely cuts the diverted chain
    """
    NUM_DOF = 10
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_div_chain=True
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

    dummy_ref_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(27),
    )

    loss = div_rollout_config(dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp)

    def manual_loss_fn(model):
        ic = dummy_data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(dummy_ref_mlp)(pred)
            ref = jax.lax.stop_gradient(ref)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

        return loss

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(div_rollout_config)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp
    )
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)

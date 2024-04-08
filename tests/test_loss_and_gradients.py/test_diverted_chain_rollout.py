import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _utils import compare_pytree

import trainax as tx


def _run(div_rollout_config, manual_loss_fn):
    NUM_DOF = 10

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]

    # using plenty of rollout to cover all potential scenarios
    shape = (5, 100, NUM_DOF)
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

    manual_loss = manual_loss_fn(dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(div_rollout_config)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp
    )
    manual_grad = eqx.filter_grad(manual_loss_fn)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp
    )

    compare_pytree(grad, manual_grad)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChainBranchOne(num_rollout_steps)

    def manual_loss_fn(model, data, *, ref_stepper):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

        return loss

    _run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration_cut_bptt(num_rollout_steps):
    """
    Completely cuts the backpropagation through time
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_bptt=True
    )

    def manual_loss_fn(model, data, *, ref_stepper):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

            pred = jax.lax.stop_gradient(pred)

        return loss

    _run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_rollout_configuration_cut_div_chain(num_rollout_steps):
    """
    Completely cuts the diverted chain
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_div_chain=True
    )

    def manual_loss_fn(model, data, *, ref_stepper):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            ref = jax.lax.stop_gradient(ref)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

        return loss

    _run(div_rollout_config, manual_loss_fn)

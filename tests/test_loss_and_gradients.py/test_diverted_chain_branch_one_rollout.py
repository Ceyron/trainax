import jax
import jax.numpy as jnp
import pytest
from _utils import run

import trainax as tx


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_branch_one_rollout_configuration(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChainBranchOne(num_rollout_steps)

    def manual_loss_fn(model, data, *, ref_stepper, residuum_fn=None):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

        return loss

    run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_branch_one_rollout_configuration_cut_bptt(num_rollout_steps):
    """
    Completely cuts the backpropagation through time
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_bptt=True
    )

    def manual_loss_fn(model, data, *, ref_stepper, residuum_fn=None):
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

    run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_branch_one_rollout_configuration_cut_div_chain(
    num_rollout_steps,
):
    """
    Completely cuts the diverted chain
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_div_chain=True
    )

    def manual_loss_fn(model, data, *, ref_stepper, residuum_fn=None):
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

    run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_diverted_chain_branch_one_rollout_configuration_cut_bptt_and_div_chain(
    num_rollout_steps,
):
    """
    Completely cuts the backpropagation through time and the diverted chain
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_bptt=True, cut_div_chain=True
    )

    def manual_loss_fn(model, data, *, ref_stepper, residuum_fn=None):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            ref = jax.lax.stop_gradient(ref)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

            pred = jax.lax.stop_gradient(pred)

        return loss

    run(div_rollout_config, manual_loss_fn)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5, 7, 8, 9])
def test_diverted_chain_branch_one_rollout_configuration_sparse_bptt(num_rollout_steps):
    """
    Cuts the bptt every three steps
    """
    div_rollout_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps, cut_bptt=True, cut_bptt_every=3
    )

    def manual_loss_fn(model, data, *, ref_stepper, residuum_fn=None):
        ic = data[:, 0, :]
        pred = ic
        loss = 0.0
        for i in range(num_rollout_steps):
            # Ref must be computed based on previous pred
            ref = jax.vmap(ref_stepper)(pred)
            pred = jax.vmap(model)(pred)
            loss += jnp.mean(jnp.square(pred - ref))

            if (i + 1) % 3 == 0:
                pred = jax.lax.stop_gradient(pred)

        return loss

    run(div_rollout_config, manual_loss_fn)

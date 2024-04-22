import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest


def compare_pytree(pytree_1, pytree_2, abs=1e-6):
    for a, b in zip(
        jtu.tree_leaves(eqx.filter(pytree_1, eqx.is_array)),
        jtu.tree_leaves(eqx.filter(pytree_2, eqx.is_array)),
    ):
        assert a.shape == b.shape
        assert a == pytest.approx(b, abs=abs)


def run(loss_config, manual_loss_fn):
    NUM_DOF = 10

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]

    shape = (5, NUM_DOF)
    dummy_ic = jax.random.normal(jax.random.PRNGKey(0), shape)

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

    dummy_res_processor_next = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(31),
    )
    dummy_res_processor_prev = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(77),
    )

    dummy_data = [
        dummy_ic,
    ]
    # Using plenty of time steps to cover the whole trajectory
    for t in range(20):
        dummy_data.append(jax.vmap(dummy_ref_mlp)(dummy_data[-1]))
    dummy_data = jnp.stack(dummy_data, axis=1)

    def residuum_fn(next, prev):
        return jnp.mean(
            jnp.square(dummy_res_processor_next(next) - dummy_res_processor_prev(prev))
        )

    loss = loss_config(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp, residuum_fn=residuum_fn
    )

    manual_loss = manual_loss_fn(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp, residuum_fn=residuum_fn
    )

    assert loss == pytest.approx(manual_loss, abs=1e-6)

    grad = eqx.filter_grad(loss_config)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp, residuum_fn=residuum_fn
    )
    manual_grad = eqx.filter_grad(manual_loss_fn)(
        dummy_mlp, dummy_data, ref_stepper=dummy_ref_mlp, residuum_fn=residuum_fn
    )

    compare_pytree(grad, manual_grad)

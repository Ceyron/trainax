import equinox as eqx
import jax
import jax.numpy as jnp
from _utils import compare_pytree

import trainax as tx


def test_one_step_supervised_configuration():
    NUM_DOF = 10
    one_step_sup_config = tx.configuration.Supervised(1)

    # (batch_size, num_rollout_steps+1, num_dof)  [not using conv format with channels here]
    shape = (5, 2, NUM_DOF)
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), shape)

    dummy_mlp = eqx.nn.MLP(
        in_size=NUM_DOF,
        out_size=NUM_DOF,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(11),
    )

    loss = one_step_sup_config(dummy_mlp, dummy_data)

    manual_loss_fn = lambda model: jnp.mean(
        jnp.square(jax.vmap(model)(dummy_data[:, 0, :]) - dummy_data[:, 1, :])
    )

    manual_loss = manual_loss_fn(dummy_mlp)

    assert manual_loss == loss

    grad = eqx.filter_grad(one_step_sup_config)(dummy_mlp, dummy_data)
    manual_grad = eqx.filter_grad(manual_loss_fn)(dummy_mlp)

    compare_pytree(grad, manual_grad)

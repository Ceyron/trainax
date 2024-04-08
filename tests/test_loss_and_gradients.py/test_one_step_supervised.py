import jax
import jax.numpy as jnp
from _utils import run

import trainax as tx


def test_one_step_supervised_configuration():
    one_step_sup_config = tx.configuration.Supervised(1)

    def manual_loss_fn(model, data, *, ref_stepper=None, residuum_fn=None):
        return jnp.mean(jnp.square(jax.vmap(model)(data[:, 0, :]) - data[:, 1, :]))

    run(one_step_sup_config, manual_loss_fn)

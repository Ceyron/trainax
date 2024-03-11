import jax.numpy as jnp
import optax

import trainax


def test_instantiate_trainers():
    data_trjs = jnp.zeros((10, 5, 1, 30))
    ref_stepper = lambda x: x
    residuum_fn = lambda x, y: x - y
    optimizer = optax.sgd(1e-3)

    trainax.trainer.SupervisedTrainer(
        data_trjs,
        optimizer=optimizer,
        num_training_steps=100,
        batch_size=10,
    )
    trainax.trainer.DivertedChainBranchOneTrainer(
        data_trjs,
        ref_stepper=ref_stepper,
        residuum_fn=residuum_fn,
        optimizer=optimizer,
        num_training_steps=100,
        batch_size=10,
    )
    trainax.trainer.ResiduumTrainer(
        data_trjs,
        residuum_fn=residuum_fn,
        optimizer=optimizer,
        num_training_steps=100,
        batch_size=10,
    )

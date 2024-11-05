# Getting started

## Installation

Clone the repository, navigate to the folder and install the package with pip:
```bash
pip install trainax
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quickstart

Train a kernel size 2 linear convolution (no bias) to become an emulator for the
1D advection problem.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import trainax as tx

CFL = -0.75

ref_data = tx.sample_data.advection_1d_periodic(
    cfl = CFL,
    key = jax.random.PRNGKey(0),
)

linear_conv_kernel_2 = eqx.nn.Conv1d(
    1, 1, 2,
    padding="SAME", padding_mode="CIRCULAR", use_bias=False,
    key=jax.random.PRNGKey(73)
)

sup_1_trainer, sup_5_trainer, sup_20_trainer = (
    tx.trainer.SupervisedTrainer(
        ref_data,
        num_rollout_steps=r,
        optimizer=optax.adam(1e-2),
        num_training_steps=1000,
        batch_size=32,
    )
    for r in (1, 5, 20)
)

sup_1_conv, sup_1_loss_history = sup_1_trainer(
    linear_conv_kernel_2, key=jax.random.PRNGKey(42)
)
sup_5_conv, sup_5_loss_history = sup_5_trainer(
    linear_conv_kernel_2, key=jax.random.PRNGKey(42)
)
sup_20_conv, sup_20_loss_history = sup_20_trainer(
    linear_conv_kernel_2, key=jax.random.PRNGKey(42)
)

FOU_STENCIL = jnp.array([1+CFL, -CFL])

print(jnp.linalg.norm(sup_1_conv.weight - FOU_STENCIL))   # 0.033
print(jnp.linalg.norm(sup_5_conv.weight - FOU_STENCIL))   # 0.025
print(jnp.linalg.norm(sup_20_conv.weight - FOU_STENCIL))  # 0.017
```

Increasing the supervised unrolling steps during training makes the learned
stencil come closer to the numerical FOU stencil.

## Features

* Wide collection of unrolled training methodologies:
    * Supervised
    * Diverted Chain
    * Mix Chain
    * Residuum
* Based on [JAX](https://github.com/google/jax):
    * One of the best Automatic Differentiation engines (forward & reverse)
    * Automatic vectorization
    * Backend-agnostic code (run on CPU, GPU, and TPU)
* Build on top and compatible with [Equinox](https://github.com/patrick-kidger/equinox)
* Batch-Parallel Training
* Collection of Callbacks
* Composability

## Acknowledgements

### Citation

This package was developed as part of the [APEBench paper
(arxiv.org/abs/2411.00180)](https://arxiv.org/abs/2411.00180) (accepted at
Neurips 2024). If you find it useful for your research, please consider citing
it:

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A Benchmark for Autoregressive Neural Emulators of {PDE}s},
  author={Felix Koehler and Simon Niedermayr and R{\"}udiger Westermann and Nils Thuerey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2024}
}
```

(Feel free to also give the project a star on GitHub if you like it.)

[Here](https://github.com/tum-pbs/apebench) you can find the APEBench benchmark
suite.

### Funding

The main author (Felix Koehler) is a PhD student in the group of [Prof. Thuerey at TUM](https://ge.in.tum.de/) and his research is funded by the [Munich Center for Machine Learning](https://mcml.ai/).

### License

MIT, see [here](https://github.com/Ceyron/trainax/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler) &nbsp;&middot;&nbsp;
> LinkedIn [Felix Köhler](https://www.linkedin.com/in/felix-koehler)

<h1 align="center">
  <img src="docs/imgs/trainax_logo.png" width="400">
  <br>
    Trainax
  <br>
</h1>

<p align="center">
<b>Learning Methodologies for Autoregressive Neural Emulators.</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#background">Background</a> •
  <a href="#features">Features</a> •
  <a href="#a-taxonomy-of-training-methodologies">Taxonomy</a> •
  <a href="#license">License</a>
</p>

<p align="center">
    <img src="docs/imgs/sample_two_step_supervised.svg" width="400">
</p>

## Installation

Clone the repository, navigate to the folder and install the package with pip:
```bash
pip install .
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quickstart

Train a kernel size 2 linear convolution (no bias) to become an emulator for the
1D advection problem.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # pip install optax
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

## Background

After the discretization of space and time, the simulation of a time-dependent
partial differential equation amounts to the repeated application of a
simulation operator $\mathcal{P}_h$. Here, we are interested in
imitating/emulating this physical/numerical operator with a neural network
$f_\theta$. This repository is concerned with an abstract implementation of all
ways we can frame a learning problem to inject "knowledge" from $\mathcal{P}_h$
into $f_\theta$.

Assume we have a distribution of initial conditions $\mathcal{Q}$ from which we
sample $S$ initial conditions, $u^{[0]} \propto \mathcal{Q}$. Then, we can save
them in an array of shape $(S, C, *N)$ (with C channels and an arbitrary number
of spatial axes of dimension N) and repeatedly apply $\mathcal{P}$ to obtain the
training trajectory of shape $(S, T+1, C, *N)$.

For a one-step supervised learning task, we substack the training trajectory
into windows of size $2$ and merge the two leftover batch axes to get a data
array of shape $(S \cdot T, 2, N)$ that can be used in supervised learning
scenario

$$
L(\theta) = \mathbb{E}_{(u^{[0]}, u^{[1]}) \sim \mathcal{Q}} \left[ l\left( f_\theta(u^{[0]}), u^{[1]} \right) \right]
$$

where $l$ is a **time-level loss**. In the easiest case $l = \text{MSE}$.

`Trainax` supports way more than just one-step supervised learning, e.g., to
train with unrolled steps, to include the reference simulator $\mathcal{P}_h$ in
training, train on residuum conditions instead of resolved reference states, cut
and modify the gradient flow, etc.

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


## A Taxonomy of Training Methodologies

The major axes that need to be chosen are:

* The unrolled length (how often the network is applied autoregressively on the
  input)
* The branch length (how long the reference goes alongside the network; we get
  full supervised if that is as long as the rollout length)
* Whether the physics is resolved (diverted-chain and supervised) or only given
  as a condition (residuum-based loss)

Additional axes are:

* The time level loss (how two states are compared, or a residuum state is reduced)
* The time level weights (if there is network rollout, shall states further away
  from the initial condition be weighted differently (like exponential
  discounting in reinforcement learning))
* If the main chain of network rollout is interleaved with a physics solver (-> mix chain)
* Modifications to the gradient flow:
    * Cutting the backpropagation through time in the main chain (after each
      step, or sparse)
    * Cutting the diverted physics
    * Cutting the one or both levels of the inputs to a residuum function.

### Implementation details

There are three levels of hierarchy:

1. The `loss` submodule defines time-level wise comparisons between two states.
   A state is either a tensor of shape `(num_channels, ...)` (with ellipsis
   indicating an arbitrary number of spatial dim,ensions) or a tensor of shape
   `(num_batches, num_channels, ...)`. The time level loss is implemented for
   the former but allows additional vectorized and (mean-)aggregated on the
   latter. (In the schematic above, the time-level loss is the green circle).
2. The `configuration` submodule devises how neural time stepper $f_\theta$
   (denoted *NN* in the schematic) interplays with the numerical simulator
   $\mathcal{P}_h$. Similar to the time-level loss this is a callable PyTree
   which requires during calling the neural stepper and some data. What this
   data contains depends on the concrete configuration. For supervised rollout
   training it is the batch of (sub-) trajectories to be considered. Other
   configurations might also require the reference stepper or a two consecutive
   time level based residuum function. Each configuration is essentially an
   abstract implementation of the major methodologies (supervised,
   diverted-chain, mix-chain, residuum). The most general diverted chain
   implementation contains supervised and branch-one diverted chain as special
   cases. All configurations allow setting additional constructor arguments to,
   e.g., cut the backpropagation through time (sparsely) or to supply time-level
   weightings (for example to exponentially discount contributions over long
   rollouts).
3. The `training` submodule combines a configuration together with stochastic
   minibatching on a set of reference trajectories. For each configuration,
   there is a corresponding trainer that essentially is sugarcoating around
   combining the relevant configuration with the `GeneralTrainer` and a
   trajectory substacker.

You can find an overview of predictor learning setups
[here](https://fkoehler.site/predictor-learning-setups/).

## License

MIT, see [here](LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler)


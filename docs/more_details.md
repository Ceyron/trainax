# More Details

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
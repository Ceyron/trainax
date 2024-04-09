# Learning Methodologies for autoregressive neural operators

![](https://ceyron.github.io/predictor-learning-setups/sup-3-none-true-full_gradient.svg)

After the discretization of space and time, the simulation of a transient
partial differential equation amounts to the repeated application of a
simulation operator $\mathcal{P}$. Here, we are interested in
imitating/emulating this physical/numerical operator with a neural network
$f_\theta$. This repository is concerned with an abstract implementation of all
ways we can frame a learning problem to inject "knowledge" from $\mathcal{P}$ to
$f_\theta$.

### Example

Consider the 1d heat equation with periodic boundary conditions on the unit
interval

$$
\begin{aligned}
\partial_t u(x, t) &= \partial_{xx} u(x, t) \\
u(0, t) &= u(1, t) \\
\end{aligned}
$$

Discretizing the domain into $N$ degrees of freedom allows to find the
forward-in-time central-in-space time stepper ([FTCS
scheme](https://en.wikipedia.org/wiki/FTCS_scheme))

$$
u_i^{[t+1]} = u_i^{[t]} + \frac{\Delta t}{(\Delta x)^2} (u_{i+1}^{[t]} - 2 u_i^{[t]} + u_{i-1}^{[t]})
$$

which we can readily implement a simulation operator $\mathcal{P}$

```python
class FTCS:
    dt: float = 0.001
    dx: float = 0.01

    def __call__(self, u):
        u_next = u + self.dt / self.dx**2 * (np.roll(u, 1) - 2 * u + np.roll(u, -1))
        return u_next
```

Now assume we have a distribution of initial conditions $\mathcal{Q}$ from which
we sample $S$ initial conditions, $u^{[0]} \propto \mathcal{Q}$. Then, we can
save them in an array of shape $(S, N)$ and repeatedly apply $\mathcal{P}$ to
obtain the training trajectory of shape $(S, T+1, N)$.

For a one-step supervised learning task, we substack the training trajectory
into windows of size $2$ and merge the two leftover batch axes to get a data
array of shape $(S \cdot T, 2, N)$ that can be used in supervised learning
scenario

$$
L(\theta) = \mathbb{E}_{(u^{[0]}, u^{[1]}) \sim \mathcal{Q}} \left[ l\left( f_\theta(u^{[0]}), u^{[1]} \right) \right]
$$

where $l$ is a **time-level loss**. In the easiest case $l = \text{MSE}$.

### More

Focus is clearly on the number of update steps, not on the number of epochs


### A taxonomy of learning setups

The major axes that need to be chosen are:

* The rollout length (how often the network is applied autoregressively on the input)
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
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _random_truncated_fourier_series_1d(
    num_points: int,
    highest_mode: int,
    *,
    no_offset: bool = True,
    key: PRNGKeyArray,
):
    s_key, c_key, o_key = jax.random.split(key, 3)

    sine_amplitudes = jax.random.uniform(
        s_key, shape=(highest_mode,), minval=-1.0, maxval=1.0
    )
    cosine_amplitudes = jax.random.uniform(
        c_key, shape=(highest_mode,), minval=-1.0, maxval=1.0
    )
    if no_offset:
        offset = 0.0
    else:
        offset = jax.random.uniform(o_key, shape=(), minval=-0.5, maxval=0.5)

    grid = jnp.linspace(0, 2 * jnp.pi, num_points + 1)[:-1]

    u_0 = offset + sum(
        a * jnp.sin((i + 1) * grid) + b * jnp.cos((i + 1) * grid)
        for i, (a, b) in enumerate(zip(sine_amplitudes, cosine_amplitudes))
    )

    return u_0


def _advect_analytical(
    u,
    *,
    cfl: float,
):
    """
    Fourier-spectral timestepper for the advection equation in 1D.

    Exact if the the state is bandlimited.
    """
    num_points = u.shape[-1]
    normalized_advection_speed = cfl / num_points
    wavenumbers = jnp.fft.rfftfreq(num_points) * num_points * 2 * jnp.pi
    u_hat = jnp.fft.rfft(u)
    u_hat_advected = u_hat * jnp.exp(-1j * wavenumbers * normalized_advection_speed)
    u_advected = jnp.fft.irfft(u_hat_advected, n=num_points)
    return u_advected


def advection_1d_periodic(
    num_points: int = 30,
    num_samples: int = 20,
    *,
    cfl: float = 0.75,
    highest_init_mode: int = 5,
    temporal_horizon: int = 100,
    key: PRNGKeyArray,
) -> Float[Array, "num_samples temporal_horizon 1 num_points"]:
    """
    Produces a reference trajectory of the simulation of 1D advection with
    periodic boundary conditions. The solution is exact due to a Fourier
    spectral solver (requires `highest_init_mode` < `num_points//2`).

    **Arguments**:

    - `num_points`: The number of grid points.
    - `num_samples`: The number of samples to generate, i.e., how many different
        trajectories.
    - `cfl`: The Courant-Friedrichs-Lewy number.
    - `highest_init_mode`: The highest mode of the initial condition.
    - `temporal_horizon`: The number of timesteps to simulate.
    - `key`: The random key.

    **Returns**:

    - A tensor of shape `(num_samples, temporal_horizon, 1, num_points)`. The
        singleton axis is to represent one channel to have format suitable for
        convolutional networks.
    """
    init_keys = jax.random.split(key, num_samples)

    u_0 = jax.vmap(
        lambda k: _random_truncated_fourier_series_1d(
            num_points, highest_init_mode, key=k
        )
    )(init_keys)

    def scan_fn(u, _):
        u_next = _advect_analytical(u, cfl=cfl)
        return u_next, u

    def rollout(init):
        _, u_trj = jax.lax.scan(scan_fn, init, jnp.arange(temporal_horizon))
        return u_trj

    u_trj = jax.vmap(rollout)(u_0)

    u_trj_with_singleton_channel = u_trj[..., None, :]

    return u_trj_with_singleton_channel


def _step_rk4(
    fn: Callable,
    u_init: Float[Array, "dof ..."],
    dt: float,
) -> Float[Array, "dof ..."]:
    k1 = fn(u_init)
    k2 = fn(u_init + 0.5 * dt * k1)
    k3 = fn(u_init + 0.5 * dt * k2)
    k4 = fn(u_init + dt * k3)
    return u_init + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _lorenz_rhs(
    u: Float[Array, "3"],
    *,
    sigma: float,
    rho: float,
    beta: float,
) -> Float[Array, "3"]:
    x, y, z = u
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return jnp.array([x_dot, y_dot, z_dot])


def make_lorenz_stepper_rk4(
    dt: float = 0.01,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> Callable[[Float[Array, "3"]], Float[Array, "3"]]:
    r"""
    Produces a timestepper for the Lorenz system using a fixed-size Runge-Kutta
    4th order scheme.

    **Arguments**:

    - `dt`: The timestep size. Depending on the values of `sigma`, `rho`, and
        `beta`, the system might be hard to integrate. Usually, a time step
        $\Delta t \in [0.01, 0.1]$ is a good choice. The default is `0.01` which
        matches https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2
    - `sigma`: The $\sigma$ parameter of the Lorenz system. The default is `10.0`.
    - `rho`: The $\rho$ parameter of the Lorenz system. The default is `28.0`.
    - `beta`: The $\beta$ parameter of the Lorenz system. The default is `8.0/3.0`.

    **Returns**:

    - A function that takes a state vector of shape `(3,)` and returns the next
        state vector of shape `(3,)`.
    """
    lorenz_rhs_params_fixed = lambda u: _lorenz_rhs(u, sigma=sigma, rho=rho, beta=beta)
    lorenz_stepper = lambda u: _step_rk4(lorenz_rhs_params_fixed, u, dt=dt)
    return lorenz_stepper


def lorenz_rk4(
    num_samples: int = 20,
    *,
    temporal_horizon: int = 1000,
    dt: float = 0.01,
    num_warmup_steps: int = 500,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    init_std: float = 1.0,
    key: PRNGKeyArray,
) -> Float[Array, "num_samples temporal_horizon 3"]:
    r"""
    Produces reference trajectories of the simple three-equation Lorenz system
    when integrated with a fixed-size Runge-Kutta 4th order scheme.

    $$
    \begin{aligned}
    \frac{dx}{dt} &= \sigma (y - x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= x y - \beta z
    \end{aligned}
    $$

    The initial conditions are drawn from a standard normal distribution for
    each of the three variables with a prescribed standard deviation (mean is
    zero).

    **Arguments**:

    - `num_samples`: The number of samples to generate, i.e., how many different
        trajectories.
    - `temporal_horizon`: The number of timesteps to simulate.
    - `dt`: The timestep size. Depending on the values of `sigma`, `rho`, and
        `beta`, the system might be hard to integrate. Usually, a time step
        $\Delta t \in [0.01, 0.1]$ is a good choice.
    - `num_warmup_steps`: The number of steps to discard from the beginning of
        the trajectory.
    - `sigma`: The $\sigma$ parameter of the Lorenz system.
    - `rho`: The $\rho$ parameter of the Lorenz system.
    - `beta`: The $\beta$ parameter of the Lorenz system.
    - `init_std`: The standard deviation of the initial conditions.
    - `key`: The random key.

    **Returns**:

    - A tensor of shape `(num_samples, temporal_horizon, 3)`.
    """

    u_0_set = jax.random.normal(key, shape=(num_samples, 3)) * init_std

    # lorenz_rhs_params_fixed = lambda u: _lorenz_rhs(u, sigma=sigma, rho=rho, beta=beta)
    # lorenz_stepper = lambda u: _step_rk4(lorenz_rhs_params_fixed, u, dt=dt)

    lorenz_stepper = make_lorenz_stepper_rk4(dt=dt, sigma=sigma, rho=rho, beta=beta)

    def scan_fn(u, _):
        u_next = lorenz_stepper(u)
        return u_next, u

    def rollout(init):
        _, u_trj = jax.lax.scan(
            scan_fn, init, None, length=temporal_horizon + num_warmup_steps
        )
        return u_trj

    trj_set = jax.vmap(rollout)(u_0_set)

    # Slice away the warmup steps
    trj_set = trj_set[:, num_warmup_steps:]

    return trj_set

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

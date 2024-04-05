import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree


def extract_ic_and_trj(data):
    ic = jtu.tree_map(lambda x: x[:, 0], data)
    trj = jtu.tree_map(lambda x: x[:, 1:], data)

    return ic, trj


def stack_sub_trajectories(
    trj: PyTree[Float[Array, "n_timesteps ..."]],
    sub_len: int,
) -> PyTree[Float[Array, "n_stacks sub_len ..."]]:
    """
    Slice a trajectory into subtrajectories of length `n` and stack them
    together. Useful for rollout training neural operators with temporal mixing.

    !!! Note that this function can produce very large arrays.

    **Arguments:**
        - `trj`: The trajectory to slice. Expected shape: `(n_timesteps, ...)`.
        - `sub_len`: The length of the subtrajectories. If you want to perform rollout
            training with k steps, note that `n=k+1` to also have an initial
            condition in the subtrajectories.

    **Returns:**
        - `sub_trjs`: The stacked subtrajectories. Expected shape: `(n_stacks,
            n, ...)`. `n_stacks` is the number of subtrajectories stacked
            together, i.e., `n_timesteps - n + 1`.
    """
    n_time_steps = [leaf.shape[0] for leaf in jtu.tree_leaves(trj)]

    if len(set(n_time_steps)) != 1:
        raise ValueError(
            "All arrays in trj must have the same number of time steps in the leading axis"
        )
    else:
        n_time_steps = n_time_steps[0]

    if sub_len > n_time_steps:
        raise ValueError(
            "n must be smaller than or equal to the number of time steps in trj"
        )

    n_sub_trjs = n_time_steps - sub_len + 1

    def scan_fn(_, i):
        sliced = jtu.tree_map(
            lambda leaf: jax.lax.dynamic_slice_in_dim(
                leaf,
                start_index=i,
                slice_size=sub_len,
                axis=0,
            ),
            trj,
        )
        return _, sliced

    _, sub_trjs = jax.lax.scan(scan_fn, None, jnp.arange(n_sub_trjs))

    return sub_trjs

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from jaxtyping import Array, Float, PyTree, PRNGKeyArray, Int

import equinox as eqx

from .utils import stack_sub_trajectories

class TrajectoryMixer(eqx.Module):
    data_sub_trajectories: PyTree[Float[Array, "num_total_samples sub_trj_len ..."]]

    num_total_samples: int
    num_minibatches: int
    batch_size: int
    num_minibatches_per_epoch: int
    num_epochs: int

    permutations: Array

    def __init__(
        self,
        data_trajectories: PyTree[Float[Array, "num_samples trj_len ..."]],
        *,
        sub_trajectory_len: int,
        num_minibatches: int,
        batch_size: int,
        do_sub_stacking: bool = True,
        only_store_ic: bool = False,
        shuffle_key: PRNGKeyArray,
    ):
        if do_sub_stacking:
            # return shape is (num_samples, num_stacks, sub_trj_len, ...)
            stacked_sub_trajectories = jax.vmap(
                stack_sub_trajectories,
                in_axes=(0, None),
            )(data_trajectories, sub_trajectory_len)
        else:
            stacked_sub_trajectories = jtu.tree_map(lambda x: x[:, :sub_trajectory_len], data_trajectories)

        # Merge the two batch axes (num_samples & num_stacks) into (num_total_samples)
        sub_trajecories = jtu.tree_map(jnp.concatenate, stacked_sub_trajectories)

        if only_store_ic:
            sub_trajecories = jtu.tree_map(lambda x: x[:, 0:1], sub_trajecories)

        num_total_samples = jtu.tree_map(lambda x: x.shape[0], (sub_trajecories,))[0]

        if num_total_samples < batch_size:
            print(f"batch size {batch_size} is larger than the total number of samples after sub stacking {num_total_samples}")
            print("Performing full batch training")
            effective_batch_size = num_total_samples
        else:
            effective_batch_size = batch_size

        self.num_total_samples = num_total_samples
        self.num_minibatches = num_minibatches
        self.num_minibatches_per_epoch = int(jnp.ceil(num_total_samples / effective_batch_size))
        self.num_epochs = int(jnp.ceil(num_minibatches / self.num_minibatches_per_epoch))
        self.batch_size = effective_batch_size

        self.data_sub_trajectories = sub_trajecories

        # Precompute the permutations
        _, self.permutations = jax.lax.scan(
            lambda key, _: (jax.random.split(key)[0], jax.random.permutation(key, num_total_samples)),
            shuffle_key,
            None,
            length=self.num_epochs,
        )

    def __call__(
        self,
        i: int,
        *,
        return_info: bool = False,
    ):
        """
        `i` the batch index
        """
        if i >= self.num_minibatches:
            raise ValueError("Batch index out of range")

        epoch_i = i // self.num_minibatches_per_epoch
        batch_i = i % self.num_minibatches_per_epoch

        batch_start = batch_i * self.batch_size
        batch_end = min((batch_i + 1) * self.batch_size, self.num_total_samples)

        batch_indices = self.permutations[epoch_i, batch_start:batch_end]
        data = jtu.tree_map(lambda x: x[batch_indices], self.data_sub_trajectories)

        if return_info:
            return data, (epoch_i, batch_i)
        else:
            return data
        

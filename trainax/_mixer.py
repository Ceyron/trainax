import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from ._utils import stack_sub_trajectories


class TrajectorySubStacker(eqx.Module):
    data_sub_trajectories: PyTree[Float[Array, "num_total_samples sub_trj_len ..."]]

    num_total_samples: int

    def __init__(
        self,
        data_trajectories: PyTree[Float[Array, "num_samples trj_len ..."]],
        sub_trajectory_len: int,
        *,
        do_sub_stacking: bool = True,
        only_store_ic: bool = False,
    ):
        """
        Slice a batch of trajectories into sub-trajectories.
        
        Useful to create windows of specific length for (rollout) training
        methodologies of autoregressive neural emulators.

        Args:
            data_trajectories (PyTree[Float[Array, "num_samples trj_len ..."]]):
                The batch of trajectories to slice. This must be a PyTree of
                Arrays who have at least two leading axes: a batch-axis and a
                time axis. For example, the zeroth axis can be associated with
                multiple initial conditions or constitutive parameters and the
                first axis represents all temporal snapshots. A PyTree can also
                just be an array. You can provide additional leafs in the
                PyTree, e.g., for the corresponding constitutive parameters etc.
                Make sure that the emulator has the corresponding signature.
            sub_trajectory_len (int): The length of the sub-trajectories. This
                must be smaller equal to the length of the trajectories
                (`trj_len`). For rollout training with `t` steps, set this to
                `t+1` to include the necessary initial condition.
            do_sub_stacking (bool, optional): Whether to slice out all possible
                (overlapping) windows out of the `trj_len` or just slice the
                `trj_len` axis from `0:sub_trajectory_len`. Defaults to True.
            only_store_ic (bool, optional): Whether to only store the initial
                condition of the sub-trajectories. This can be helpful for
                configurations that do not need the reference trajectory like
                residuum-based learning strategies. Defaults to False.
        """
        if do_sub_stacking:
            # return shape is (num_samples, num_stacks, sub_trj_len, ...)
            stacked_sub_trajectories = jax.vmap(
                stack_sub_trajectories,
                in_axes=(0, None),
            )(data_trajectories, sub_trajectory_len)
        else:
            # shape is (num_samples, 1, sub_trj_len, ...)
            stacked_sub_trajectories = jtu.tree_map(
                lambda x: x[:, None, :sub_trajectory_len], data_trajectories
            )

        # Merge the two batch axes (num_samples & num_stacks) into (num_total_samples)
        # resulting shape is (num_total_samples, sub_trj_len, ...)
        sub_trajecories = jtu.tree_map(jnp.concatenate, stacked_sub_trajectories)

        if only_store_ic:
            # changes shape to (num_total_samples, 1, ...)
            sub_trajecories = jtu.tree_map(lambda x: x[:, 0:1], sub_trajecories)

        num_total_samples = jtu.tree_map(lambda x: x.shape[0], (sub_trajecories,))[0]

        self.num_total_samples = num_total_samples
        self.data_sub_trajectories = sub_trajecories

    def __call__(
        self,
        indices,
    ):
        """
        Slice out sub-samples based on the given indices.
        """
        return jtu.tree_map(lambda x: x[indices], self.data_sub_trajectories)


class PermutationMixer(eqx.Module):
    num_total_samples: int
    num_minibatches: int
    batch_size: int
    num_minibatches_per_epoch: int
    num_epochs: int

    permutations: Array

    def __init__(
        self,
        num_total_samples: int,
        num_minibatches: int,
        batch_size: int,
        shuffle_key: PRNGKeyArray,
    ):
        if num_total_samples < batch_size:
            print(
                f"batch size {batch_size} is larger than the total number of samples {num_total_samples}"
            )
            print("Performing full batch training")
            effective_batch_size = num_total_samples
        else:
            effective_batch_size = batch_size

        self.num_total_samples = num_total_samples
        self.num_minibatches = num_minibatches
        self.num_minibatches_per_epoch = int(
            jnp.ceil(num_total_samples / effective_batch_size)
        )
        self.num_epochs = int(
            jnp.ceil(num_minibatches / self.num_minibatches_per_epoch)
        )
        self.batch_size = effective_batch_size

        # Precompute the permutations
        _, self.permutations = jax.lax.scan(
            lambda key, _: (
                jax.random.split(key)[0],
                jax.random.permutation(key, num_total_samples),
            ),
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

        if return_info:
            return batch_indices, (epoch_i, batch_i)
        else:
            return batch_indices


class TrajectoryMixer(eqx.Module):
    """
    Convenience class to combine `TrajectorySubStacker` and `PermutationMixer`
    """

    trajectory_sub_stacker: TrajectorySubStacker
    permutation_mixer: PermutationMixer

    def __init__(
        self,
        data_trajectories: PyTree[Float[Array, "num_samples trj_len ..."]],
        *,
        sub_trajectory_len: int,
        num_minibatches: int,
        batch_size: int,
        shuffle_key: PRNGKeyArray,
        do_sub_stacking: bool = True,
        only_store_ic: bool = False,
    ):
        print(
            "Please prefer using the `TrajectorySubStacker` and `PermutationMixer` directly"
        )
        self.trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len,
            do_sub_stacking=do_sub_stacking,
            only_store_ic=only_store_ic,
        )

        self.permutation_mixer = PermutationMixer(
            self.trajectory_sub_stacker.num_total_samples,
            num_minibatches,
            batch_size,
            shuffle_key,
        )

    def __call__(
        self,
        i: int,
        *,
        return_info: bool = False,
    ):
        if return_info:
            batch_indices, permutation_info = self.permutation_mixer(
                i, return_info=True
            )
            return self.trajectory_sub_stacker(batch_indices), permutation_info
        else:
            batch_indices = self.permutation_mixer(i)
            return self.trajectory_sub_stacker(batch_indices)

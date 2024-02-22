from jaxtyping import PRNGKeyArray

import equinox as eqx
from ..trainer import Trainer
from ..time_level_loss import TimeLevelLoss, L2Loss
from ..mixer import TrajectoryMixer
from ..loss_configuration import Supervised

class SupervisedTrainer(Trainer):
    def __init__(
        self,
        data_trajectories,
        *,
        ref_stepper: eqx.Module = None,  # for compatibility
        residuum_fn: eqx.Module = None,  # for compatibility
        optimizer,
        callback_fn = None,
        num_training_steps: int,
        batch_size: int,
        shuffle_key: PRNGKeyArray,
        num_rollout_steps: int = 1,
        time_level_loss: TimeLevelLoss = L2Loss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: list[float] = None,
    ):
        trajectory_mixer = TrajectoryMixer(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1, # +1 for the IC
            num_minibatches=num_training_steps,
            batch_size=batch_size,
            shuffle_key=shuffle_key,
        )
        loss_configuration = Supervised(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
            time_level_weights=time_level_weights,
        )
        super().__init__(
            trajectory_mixer,
            loss_configuration,
            ref_stepper=ref_stepper,
            residuum_fn=residuum_fn,
            optimizer=optimizer,
            callback_fn=callback_fn,
        )

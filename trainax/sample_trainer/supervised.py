import equinox as eqx

from ..loss_configuration import Supervised
from ..mixer import TrajectorySubStacker
from ..time_level_loss import L2Loss, TimeLevelLoss
from ..trainer import Trainer


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        data_trajectories,
        *,
        ref_stepper: eqx.Module = None,  # for compatibility
        residuum_fn: eqx.Module = None,  # for compatibility
        optimizer,
        callback_fn=None,
        num_training_steps: int,
        batch_size: int,
        num_rollout_steps: int = 1,
        time_level_loss: TimeLevelLoss = L2Loss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        time_level_weights: list[float] = None,
        do_sub_stacking: bool = True,
    ):
        trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1,  # +1 for the IC
            do_sub_stacking=do_sub_stacking,
            only_store_ic=False,
        )
        loss_configuration = Supervised(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
            time_level_weights=time_level_weights,
        )
        super().__init__(
            trajectory_sub_stacker,
            loss_configuration,
            ref_stepper=ref_stepper,
            residuum_fn=residuum_fn,
            optimizer=optimizer,
            num_minibatches=num_training_steps,
            batch_size=batch_size,
            callback_fn=callback_fn,
        )

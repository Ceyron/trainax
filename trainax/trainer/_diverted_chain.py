import equinox as eqx

from .._general_trainer import GeneralTrainer
from .._mixer import TrajectorySubStacker
from ..configuration import DivertedChainBranchOne
from ..loss import BaseLoss, L2Loss


class DivertedChainBranchOneTrainer(GeneralTrainer):
    def __init__(
        self,
        data_trajectories,
        *,
        ref_stepper: eqx.Module,
        residuum_fn: eqx.Module = None,  # for compatibility
        optimizer,
        callback_fn=None,
        num_training_steps: int,
        batch_size: int,
        num_rollout_steps: int = 1,
        time_level_loss: BaseLoss = L2Loss(),
        cut_bptt: bool = False,
        cut_bptt_every: int = 1,
        cut_div_chain: bool = False,
        time_level_weights: list[float] = None,
        do_sub_stacking: bool = True,
    ):
        trajectory_sub_stacker = TrajectorySubStacker(
            data_trajectories,
            sub_trajectory_len=num_rollout_steps + 1,  # +1 for the IC
            do_sub_stacking=do_sub_stacking,
            only_store_ic=True,  # Not needed because we use the ref_stepper
        )
        loss_configuration = DivertedChainBranchOne(
            num_rollout_steps=num_rollout_steps,
            time_level_loss=time_level_loss,
            cut_bptt=cut_bptt,
            cut_bptt_every=cut_bptt_every,
            cut_div_chain=cut_div_chain,
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

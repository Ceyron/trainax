"""
A trainer performs the optimization using a configuration with correct
minibatching. Internally, it combines a `BaseConfiguration` with the correctly
set up `TrajectorySubStacker` and subclasses the `BaseTrainer
"""


from ._diverted_chain_branch_one import DivertedChainBranchOneTrainer
from ._residuum import ResiduumTrainer
from ._supervised import SupervisedTrainer

__all__ = [
    "DivertedChainBranchOneTrainer",
    "ResiduumTrainer",
    "SupervisedTrainer",
]

"""
A trainer performs the optimization using a configuration with correct
minibatching.
"""


from ._diverted_chain import DivertedChainBranchOneTrainer
from ._residuum import ResiduumTrainer
from ._supervised import SupervisedTrainer

__all__ = [
    "DivertedChainBranchOneTrainer",
    "ResiduumTrainer",
    "SupervisedTrainer",
]

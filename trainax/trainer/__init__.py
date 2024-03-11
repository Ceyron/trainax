"""
A trainer performs the optimization using a configuration with correct
minibatching.
"""


from .diverted_chain import DivertedChainBranchOneTrainer
from .residuum import ResiduumTrainer
from .supervised import SupervisedTrainer

__all__ = [
    "DivertedChainBranchOneTrainer",
    "ResiduumTrainer",
    "SupervisedTrainer",
]

"""
A configuration is a compute graph that is used to train an autoregressive
neural operator.
"""


from ._base_configuration import BaseConfiguration
from ._composite import CompositeLossConfiguratoin
from ._diverted_chain_branch_one import DivertedChainBranchOne
from ._mix_chain import MixChainPostPhysics
from ._residuum import Residuum
from ._supervised import Supervised

__all__ = [
    "BaseConfiguration",
    "CompositeLossConfiguratoin",
    "DivertedChainBranchOne",
    "MixChainPostPhysics",
    "Residuum",
    "Supervised",
]

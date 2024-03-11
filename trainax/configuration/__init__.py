"""
A configuration is a compute graph that is used to train an autoregressive
neural operator.
"""


from .base_configuration import BaseConfiguration
from .composite import CompositeLossConfiguratoin
from .diverted_chain import DivertedChainBranchOne
from .mix_chain import MixChainPostPhysics
from .residuum import Residuum
from .supervised import Supervised

__all__ = [
    "BaseConfiguration",
    "CompositeLossConfiguratoin",
    "DivertedChainBranchOne",
    "MixChainPostPhysics",
    "Residuum",
    "Supervised",
]

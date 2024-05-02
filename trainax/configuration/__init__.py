"""
A configuration is a compute graph that is used to train an autoregressive
neural operator.
"""


from ._base_configuration import BaseConfiguration
from ._composite import Composite
from ._diverted_chain import DivertedChain
from ._diverted_chain_branch_one import DivertedChainBranchOne
from ._mix_chain_post_physics import MixChainPostPhysics
from ._residuum import Residuum
from ._supervised import Supervised

__all__ = [
    "BaseConfiguration",
    "Composite",
    "DivertedChain",
    "DivertedChainBranchOne",
    "MixChainPostPhysics",
    "Residuum",
    "Supervised",
]

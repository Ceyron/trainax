"""
A configuration is a compute graph that is used to train an autoregressive
neural emulator. It can be purely data-driven like supervised (rollout) training
(`Supervised`) allowing the data-generator to be turned off during training.
Alternatively, the emulator and either the `ref_stepper` (used to create the
data) or the `residuum_fn` (a discrete condition on two consecutive time steps)
can be more deeply interwoven. A `ref_stepper` is needed for the `DivertedChain`
configuration and its special case `DivertedChainBranchOne` as well as the
`MixChainPostPhysics`. The `Residuum` configuration uses a `residuum_fn` to
compute the loss.

The `Composite` configuration allows to combine multiple configurations with
respective weights.
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

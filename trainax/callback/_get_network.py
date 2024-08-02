from typing import Any

import equinox as eqx
from jaxtyping import PyTree

from ._base import BaseCallback


class GetNetwork(BaseCallback):
    def __init__(self, every: int, name: str = "network"):
        """Callback to write out the network state `every` update step."""
        super().__init__(every, name)

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Any:
        """Write out the network state."""
        return stepper

from typing import Any

import equinox as eqx
from jaxtyping import PyTree

from ._base import BaseCallback


class GetNetwork(BaseCallback):
    def __init__(self, every: int, name: str = "network"):
        self.every = every
        self.name = name

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Any:
        return stepper

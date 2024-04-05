from typing import Any

import equinox as eqx
from jaxtyping import PyTree

from ._base import BaseCallback


class SaveNetwork(BaseCallback):
    path: str
    file_name: str

    def __init__(
        self,
        every: int,
        path: str,
        file_name: str,
        name: str = "network_saved",
    ):
        self.every = every
        self.path = path
        self.file_name = file_name
        self.name = name

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Any:
        concrete_file_name = f"{self.path}/{self.file_name}_{update_i}.eqx"
        eqx.tree_serialise_leaves(stepper, concrete_file_name)
        return True

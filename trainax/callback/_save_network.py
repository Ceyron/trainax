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
        """
        Callback to write the network state to a file `every` update step.

        **Arguments:**

        - `every`: The frequency of the callback.
        - `path`: The path to save the network state.
        - `file_name`: The file name to save the network state.
        - `name`: The name of the callback
        """
        self.path = path
        self.file_name = file_name
        super().__init__(every, name)

    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Any:
        concrete_file_name = f"{self.path}/{self.file_name}_{update_i}.eqx"
        eqx.tree_serialise_leaves(stepper, concrete_file_name)
        return True

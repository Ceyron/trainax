from abc import ABC, abstractmethod
from typing import Any, Dict

import equinox as eqx
from jaxtyping import PyTree


class BaseCallback(eqx.Module, ABC):
    every: int
    name: str

    def __init__(self, every: int, name: str):
        """Base class for callbacks."""
        self.every = every
        self.name = name

    @abstractmethod
    def callback(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ):
        pass

    def __call__(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Dict[str, Any]:
        """
        Evaluate the Callback.

        **Arguments:**

        - `update_i`: The current update step.
        - `stepper`: The equinox.Module to evaluate the callback on.
        - `data`: The data to evaluate the callback on.

        **Returns:**

        - The result of the callback wrapped into a dictionary.
        """
        if update_i % self.every == 0:
            res = self.callback(update_i, stepper, data)
            return {self.name: res}
        else:
            return {self.name: None}

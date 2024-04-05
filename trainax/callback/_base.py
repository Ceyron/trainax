from abc import ABC, abstractmethod
from typing import Any, Dict

import equinox as eqx
from jaxtyping import PyTree


class BaseCallback(eqx.Module, ABC):
    every: int
    name: str

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
        if update_i % self.every == 0:
            res = self.callback(update_i, stepper, data)
            return {self.name: res}
        else:
            return {self.name: None}

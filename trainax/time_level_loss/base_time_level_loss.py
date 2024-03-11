from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array


class TimeLevelLoss(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        prediction: Array,
        target: Array,
    ):
        pass

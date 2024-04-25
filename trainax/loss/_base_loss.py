from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array


class BaseLoss(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        prediction: Array,
        target: Array,
    ) -> float:
        pass

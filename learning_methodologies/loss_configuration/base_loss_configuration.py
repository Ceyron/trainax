from abc import ABC, abstractmethod
import equinox as eqx

from ..time_level_loss import TimeLevelLoss

class LossConfiguration(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        stepper: eqx.Module,
        data,
        *,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
    ) -> float:
        pass
from equinox import Module

from .base_loss_configuration import LossConfiguration


class CompositeLossConfiguratoin(LossConfiguration):
    configurations: list[LossConfiguration]
    weights: list[float]

    def __call__(
        self,
        stepper: Module,
        data,
        *,
        ref_stepper: Module = None,
        residuum_fn: Module = None
    ) -> float:
        loss = sum(
            weight
            * conf(stepper, data, ref_stepper=ref_stepper, residuum_fn=residuum_fn)
            for conf, weight in zip(self.configurations, self.weights)
        )
        return loss

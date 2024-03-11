from equinox import Module

from ._base_configuration import BaseConfiguration


class CompositeLossConfiguratoin(BaseConfiguration):
    configurations: list[BaseConfiguration]
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

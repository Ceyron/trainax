from equinox import Module

from ._base_configuration import BaseConfiguration


class Composite(BaseConfiguration):
    configurations: list[BaseConfiguration]
    weights: list[float]

    def __init__(
        self,
        configurations: list[BaseConfiguration],
        weights: list[float],
    ):
        """
        Compose configurations with respective weights.

        **Arguments:**

        - `configurations`: The list of configurations to compose.
        - `weights`: The list of weights to apply to the configurations.
        """
        self.configurations = configurations
        self.weights = weights

    def __call__(
        self,
        stepper: Module,
        data,
        *,
        ref_stepper: Module = None,
        residuum_fn: Module = None,
    ) -> float:
        """
        Evaluate the composite configuration on the given data.

        Based on the underlying configurations, `ref_stepper` or `residuum_fn`
        or both have to be supplied (as keyword-only arguments).

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            depends on the concrete configuration. In the most reduced case, it
            just contains the set of initial states.
        - `ref_stepper`: The reference stepper to use for some
            configurations.
        - `residuum_fn`: The residuum function to use for some
            configurations.

        **Returns:**

        - The loss value computed by all configurations combined and weighted.
        """
        loss = sum(
            weight
            * conf(stepper, data, ref_stepper=ref_stepper, residuum_fn=residuum_fn)
            for conf, weight in zip(self.configurations, self.weights)
        )
        return loss

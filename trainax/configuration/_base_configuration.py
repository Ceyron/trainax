from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import PyTree


class BaseConfiguration(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        stepper: eqx.Module,
        data: PyTree,
        *,
        ref_stepper: eqx.Module = None,
        residuum_fn: eqx.Module = None,
    ) -> float:
        """
        Evaluate the configuration on the given data.

        **Arguments:**

        - `stepper`: The stepper to use for the configuration. Must
            have the signature `stepper(u_prev: PyTree) -> u_next: PyTree`.
        - `data`: The data to evaluate the configuration on. This
            depends on the concrete configuration. In the most reduced case, it
            just contains the set of initial states.
        - `ref_stepper`: The reference stepper to use for some
            configurations. (keyword-only argument)
        - `residuum_fn`: The residuum function to use for some
            configurations. (keyword-only argument)

        **Returns:**

        - The loss value computed by this configuration.
        """
        pass

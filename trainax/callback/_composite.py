from typing import Any

import equinox as eqx
from jaxtyping import PyTree

from ._base import BaseCallback


class CompositeCallback(eqx.Module):
    callbacks: list[BaseCallback]

    def __init__(self, callbacks: list[BaseCallback]):
        """Callback to combine multiple callbacks."""
        self.callbacks = callbacks

    def __call__(
        self,
        update_i: int,
        stepper: eqx.Module,
        data: PyTree,
    ) -> Any:
        res = {}
        for callback in self.callbacks:
            res.update(callback(update_i, stepper, data))
        return res

import equinox as eqx
import jax.tree_util as jtu
import pytest


def compare_pytree(pytree_1, pytree_2, abs=1e-6):
    for a, b in zip(
        jtu.tree_leaves(eqx.filter(pytree_1, eqx.is_array)),
        jtu.tree_leaves(eqx.filter(pytree_2, eqx.is_array)),
    ):
        assert a.shape == b.shape
        assert a == pytest.approx(b, abs=abs)

import pytest
from _utils import run

import trainax as tx


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_against_branch_one(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps, num_branch_steps=1
    )

    div_rollout_branch_one_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps
    )

    run(div_rollout_config, div_rollout_branch_one_config)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_against_branch_one_cut_bptt(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps, num_branch_steps=1, cut_bptt=True
    )

    div_rollout_branch_one_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps,
        cut_bptt=True,
    )

    run(div_rollout_config, div_rollout_branch_one_config)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_against_branch_one_cut_div_chain(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps, num_branch_steps=1, cut_div_chain=True
    )

    div_rollout_branch_one_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps,
        cut_div_chain=True,
    )

    run(div_rollout_config, div_rollout_branch_one_config)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_against_branch_one_cut_bptt_and_cut_div_chain(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps, num_branch_steps=1, cut_bptt=True, cut_div_chain=True
    )

    div_rollout_branch_one_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps,
        cut_bptt=True,
        cut_div_chain=True,
    )

    run(div_rollout_config, div_rollout_branch_one_config)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_against_branch_one_sparse_bptt(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps,
        num_branch_steps=1,
        cut_bptt=True,
        cut_bptt_every=3,
    )

    div_rollout_branch_one_config = tx.configuration.DivertedChainBranchOne(
        num_rollout_steps,
        cut_bptt=True,
        cut_bptt_every=3,
    )

    run(div_rollout_config, div_rollout_branch_one_config)


@pytest.mark.parametrize("num_rollout_steps", [1, 2, 3, 4, 5])
def test_against_supervised_rollout(num_rollout_steps):
    div_rollout_config = tx.configuration.DivertedChain(
        num_rollout_steps,
        num_branch_steps=num_rollout_steps,
    )

    supervised_rollout_config = tx.configuration.Supervised(num_rollout_steps)

    run(div_rollout_config, supervised_rollout_config)

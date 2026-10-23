"""Check reward attribution before collecting expensive pi0 rollouts."""

import pytest

from experiments.pi0_stove.rl import select_reward


def test_oracle_and_model_rewards_use_distinct_sources_on_the_same_chunk():
    sparse = [0.0, 0.0, 1.0]
    predicted = [0.1, 0.2, 0.3]
    assert select_reward("sparse", sparse, predicted) == 1.0
    assert select_reward("robometer", sparse, predicted) == pytest.approx(0.2)
    assert select_reward("robometer", sparse, [0.0, 0.0]) == 0.0


def test_model_reward_fails_on_missing_or_invalid_predictions():
    assert select_reward("sparse", [0.0, 0.0], None) == 0.0
    for values in (None, [], [float("nan")], [-0.01], [1.01]):
        with pytest.raises(ValueError):
            select_reward("robometer", [1.0], values)

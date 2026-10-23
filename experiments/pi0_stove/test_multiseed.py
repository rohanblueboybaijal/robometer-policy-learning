"""Checks for five-seed curve alignment and GPU-lane scheduling."""

from experiments.pi0_stove.multiseed import KINDS, LANES, NEW_SEEDS, SEEDS, aggregate


def test_every_new_seed_gets_both_rewards_once():
    jobs = [job for lane in LANES.values() for job in lane]
    assert len(jobs) == len(set(jobs)) == len(NEW_SEEDS) * len(KINDS)
    assert set(jobs) == {(kind, seed) for kind in KINDS for seed in NEW_SEEDS}
    assert SEEDS == (100, *NEW_SEEDS)


def test_aggregate_aligns_slightly_different_physical_steps():
    curves = {seed: [] for seed in SEEDS}
    curves[100] = [{"env_steps": 0, "success_rate": .4},
                   {"env_steps": 5018, "success_rate": .6}]
    curves[101] = [{"env_steps": 0, "success_rate": .2},
                   {"env_steps": 5004, "success_rate": .4}]
    points = aggregate(curves)
    assert [point["env_steps"] for point in points] == [0, 5000]
    assert [point["n"] for point in points] == [2, 2]
    assert abs(points[0]["mean"] - .3) < 1e-10
    assert abs(points[1]["mean"] - .5) < 1e-10

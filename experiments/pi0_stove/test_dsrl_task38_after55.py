"""The extra task-38 runs must wait for every task-55 reward arm."""

from experiments.pi0_stove import dsrl_task38_after55 as queue


def test_task55_gate_requires_all_five_paired_seeds(monkeypatch):
    completed = {(kind, seed) for seed in queue.PRIOR_SEEDS for kind in queue.KINDS}
    monkeypatch.setattr(queue, "run_dir", lambda task, kind, seed: (task, kind, seed))
    monkeypatch.setattr(queue, "complete", lambda run: run[0] == 55 and run[1:] in completed)
    assert queue.task55_complete()
    completed.remove(("robometer", 104))
    assert not queue.task55_complete()

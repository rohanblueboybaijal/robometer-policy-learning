"""Check the handoff and matched settings for the queued task-08 run."""

import json

from experiments.pi0_stove.dsrl_libero10_task06 import config
from experiments.pi0_stove.dsrl_libero10_task08 import SEEDS, task06_complete


def test_task08_keeps_task06_training_hyperparameters():
    for kind in ("sparse", "robometer"):
        first = config(100, kind, 0, 6)
        second = config(100, kind, 0, 8)
        for key in ("sac", "dsrl", "steps", "horizon", "eval_every", "eval_episodes",
                    "train_init_states", "save_every", "rolling_checkpoint_only",
                    "pi0_checkpoint", "reward_checkpoint"):
            assert first[key] == second[key], key
        assert second["suite"] == "libero_10" and second["task_id"] == 8


def test_handoff_requires_every_sparse_and_robometer_final_checkpoint(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "suite": "libero_10", "task_id": 6, "seeds": list(SEEDS),
    }))
    for seed in SEEDS:
        for kind in ("sparse", "robometer"):
            folder = tmp_path / f"seed{seed}" / f"{kind}_seed{seed}"
            (folder / "checkpoints").mkdir(parents=True)
            (folder / "status.json").write_text(json.dumps({"status": "complete", "env_steps": 100000}))
            if (seed, kind) != (107, "robometer"):
                (folder / "checkpoints/final.pt").write_bytes(b"checkpoint")
    assert not task06_complete(tmp_path)
    (tmp_path / "seed107/robometer_seed107/checkpoints/final.pt").write_bytes(b"checkpoint")
    assert task06_complete(tmp_path)

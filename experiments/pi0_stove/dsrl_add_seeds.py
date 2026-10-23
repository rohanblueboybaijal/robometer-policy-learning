"""Add paired direct-DSRL seeds 103 and 104 to LIBERO-90 tasks 38 and 55."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from experiments.pi0_stove.dsrl_multitask import DATA_ROOT, KINDS, TASKS, now, stage, stage_root, wait_for_orphans
from experiments.reward_audit.config import write_json


NEW_SEEDS = (103, 104)
ALL_SEEDS = (100, 101, 102, *NEW_SEEDS)
ROOT = DATA_ROOT / "dsrl_add_seeds_103_104_20261013"


def extend_manifests() -> None:
    for task_id, task_name in TASKS.items():
        path = stage_root(task_id) / "manifest.json"
        manifest = json.loads(path.read_text())
        if (manifest.get("task_id") != task_id or manifest.get("task") != task_name or
                manifest.get("algorithm") != "direct DSRL" or
                tuple(manifest["seeds"]) not in ((100, 101, 102), ALL_SEEDS)):
            raise RuntimeError(f"Unexpected existing DSRL manifest: {path}")
        manifest["seeds"] = ALL_SEEDS
        manifest["additional_seeds_source"] = str(ROOT)
        write_json(path, manifest)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Additional-seed scheduler already running: {ROOT}") from None
        manifest_path = ROOT / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("tasks") != list(TASKS) or manifest.get("seeds") != list(NEW_SEEDS):
                raise RuntimeError(f"Unexpected extension manifest: {manifest_path}")
        else:
            write_json(manifest_path, {"tasks": list(TASKS), "seeds": NEW_SEEDS,
                                       "rewards": KINDS, "steps_per_run": 100000,
                                       "algorithm": "direct DSRL", "created": now()})
        extend_manifests()
        wait_for_orphans(ROOT)
        for task_id in TASKS:
            stage(ROOT, task_id, NEW_SEEDS)
        write_json(ROOT / "orchestrator.json", {"status": "complete", "pid": os.getpid(),
                                                "updated": now()})


if __name__ == "__main__":
    main()

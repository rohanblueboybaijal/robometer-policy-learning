"""Run three more paired task-38 DSRL seeds after task-55 seeds finish."""

from __future__ import annotations

import fcntl
import json
import os
import time

from experiments.pi0_stove.dsrl_multitask import (
    DATA_ROOT, KINDS, TASKS, complete, now, run_dir, stage, stage_root, wait_for_orphans,
)
from experiments.reward_audit.config import write_json


PRIOR_SEEDS = (100, 101, 102, 103, 104)
NEW_SEEDS = (105, 106, 107)
ROOT = DATA_ROOT / "dsrl_task38_after55_105_107_20261013"
PRIOR_SCHEDULER = DATA_ROOT / "dsrl_add_seeds_103_104_20261013" / "orchestrator.json"


def task55_complete() -> bool:
    return all(complete(run_dir(55, kind, seed)) for seed in PRIOR_SEEDS for kind in KINDS)


def scheduler_alive(pid: int) -> bool:
    command_file = f"/proc/{pid}/cmdline"
    try:
        with open(command_file, "rb") as stream:
            command = stream.read().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    return "experiments.pi0_stove.dsrl_add_seeds" in command


def queue_manifest() -> None:
    path = stage_root(38) / "manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("task_id") != 38 or manifest.get("task") != TASKS[38] or
            manifest.get("algorithm") != "direct DSRL" or
            tuple(manifest["seeds"]) not in (PRIOR_SEEDS, PRIOR_SEEDS + NEW_SEEDS)):
        raise RuntimeError(f"Unexpected task-38 DSRL manifest: {path}")
    manifest["seeds"] = PRIOR_SEEDS + NEW_SEEDS
    manifest["next_seeds_source"] = str(ROOT)
    write_json(path, manifest)


def wait_for_task55() -> None:
    while not task55_complete():
        state = json.loads(PRIOR_SCHEDULER.read_text())
        if not scheduler_alive(int(state["pid"])):
            raise RuntimeError("Task-55 scheduler stopped before all seed pairs completed")
        write_json(ROOT / "orchestrator.json", {
            "status": "waiting_for_task55", "pid": os.getpid(),
            "dependency": str(PRIOR_SCHEDULER), "updated": now(),
        })
        time.sleep(30)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Task-38 extension scheduler already running: {ROOT}") from None
        path = ROOT / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("tasks") != [38] or manifest.get("seeds") != list(NEW_SEEDS):
                raise RuntimeError(f"Unexpected extension manifest: {path}")
        else:
            write_json(path, {"tasks": [38], "seeds": NEW_SEEDS, "rewards": KINDS,
                              "steps_per_run": 100000, "algorithm": "direct DSRL",
                              "after": "task 55 seeds 100-104 complete", "created": now()})
        queue_manifest()
        wait_for_orphans(ROOT)
        wait_for_task55()
        stage(ROOT, 38, NEW_SEEDS)
        write_json(ROOT / "orchestrator.json", {"status": "complete", "pid": os.getpid(),
                                                "updated": now()})


if __name__ == "__main__":
    main()

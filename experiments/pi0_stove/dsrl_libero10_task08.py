"""Start eight paired task-08 DSRL seeds after all task-06 seeds finish."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from experiments.pi0_stove.dsrl_libero10_task06 import (
    KINDS, ROOT as TASK06_ROOT, complete, initialize, run,
)
from experiments.pi0_stove.dsrl_multitask import now
from experiments.reward_audit.config import write_json


ROOT = Path("/data/rohan/Reward-Modeling/pi0_libero10_task08_dsrl_8seeds_20261016")
SEEDS = tuple(range(100, 108))


def task06_complete(root: Path = TASK06_ROOT) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("suite") != "libero_10" or manifest.get("task_id") != 6 or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError(f"Unexpected task-06 dependency manifest: {manifest_path}")
    return all(complete(seed, kind, root) for seed in SEEDS for kind in KINDS)


def task06_coordinator_alive(root: Path = TASK06_ROOT) -> bool:
    state_path = root / "orchestrator.json"
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    pid = state.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"experiments.pi0_stove.dsrl_libero10_task06_add_seeds" in command


def wait_for_task06() -> None:
    while not task06_complete():
        previous = json.loads((TASK06_ROOT / "orchestrator.json").read_text())
        if previous.get("status") == "failed" or not task06_coordinator_alive():
            raise RuntimeError("Task-06 coordinator stopped before all eight seed pairs completed")
        write_json(ROOT / "orchestrator.json", {
            "status": "waiting_for_task06", "pid": os.getpid(),
            "dependency": str(TASK06_ROOT), "updated": now(),
            "active": {}, "pending": [{"seed": seed, "kind": kind}
                                      for seed in SEEDS for kind in KINDS],
        })
        time.sleep(30)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Task-08 coordinator already running: {ROOT}") from None
        initialize(SEEDS, ROOT, 8)
        try:
            wait_for_task06()
            run(SEEDS, ROOT, 8)
        except BaseException as error:
            write_json(ROOT / "orchestrator.json", {"status": "failed", "pid": os.getpid(),
                                                    "error": repr(error), "updated": now()})
            raise


if __name__ == "__main__":
    main()

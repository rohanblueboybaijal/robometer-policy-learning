"""Extend task-06 direct DSRL comparison with matched seeds 105–107."""

from __future__ import annotations

import fcntl
import json

from experiments.pi0_stove.dsrl_libero10_task06 import (
    KINDS, ROOT, SEEDS, complete, config, run, run_dir,
)
from experiments.pi0_stove.dsrl_multitask import now
from experiments.reward_audit.config import write_json


NEW_SEEDS = (105, 106, 107)
ALL_SEEDS = SEEDS + NEW_SEEDS


def extend_manifest() -> None:
    path = ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("suite") != "libero_10" or manifest.get("task_id") != 6 or
            manifest.get("algorithm") != "direct DSRL" or
            tuple(manifest.get("seeds", [])) not in (SEEDS, ALL_SEEDS)):
        raise RuntimeError(f"Unexpected task-06 experiment manifest: {path}")
    for seed in SEEDS:
        for kind in KINDS:
            if not complete(seed, kind):
                raise RuntimeError(f"Original task-06 run is incomplete: {run_dir(seed, kind)}")
            saved = json.loads((run_dir(seed, kind) / "config.json").read_text())
            if saved != config(seed, kind, int(saved["policy_gpu"])):
                raise RuntimeError(f"Original task-06 config differs: {run_dir(seed, kind)}")
    for seed in NEW_SEEDS:
        for kind in KINDS:
            folder = run_dir(seed, kind)
            if folder.exists() and not complete(seed, kind):
                raise RuntimeError(f"Incomplete extension run needs inspection: {folder}")
    manifest["seeds"] = list(ALL_SEEDS)
    manifest["additional_seeds"] = list(NEW_SEEDS)
    manifest["additional_seeds_source"] = "experiments.pi0_stove.dsrl_libero10_task06_add_seeds"
    manifest["extended_at"] = now()
    write_json(path, manifest)


def main() -> None:
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another task-06 coordinator is running: {ROOT}") from None
        extend_manifest()
        run(ALL_SEEDS)


if __name__ == "__main__":
    main()

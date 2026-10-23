"""Refresh the task-38 report after canceling queued seeds 102-104."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.pi0_stove.task38_five_seeds import now, report
from experiments.reward_audit.config import write_json


def alive(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    request = json.loads((root / "stop_after_seed101.json").read_text())
    pids = request["active_training_pids"]
    coordinator = json.loads((root / "orchestrator.json").read_text())["pid"]

    # The old coordinator owns the report and status files until it exits.
    while alive(coordinator):
        time.sleep(2)

    while True:
        states = []
        for seed, pid in zip((100, 101), pids, strict=True):
            run = root / f"seed{seed}" / f"robometer_seed{seed}"
            status_file = run / "status.json"
            state = json.loads(status_file.read_text()) if status_file.exists() else {}
            finished = (state.get("status") == "complete"
                        and state.get("env_steps") == 100000
                        and (run / "checkpoints/final.pt").exists())
            states.append((seed, finished, alive(pid)))

        report(root)
        if all(finished for _, finished, _ in states):
            status = "complete_limited"
        elif any(not finished and not running for _, finished, running in states):
            status = "failed_current_run"
        else:
            status = "finishing_current_runs"
        write_json(root / "orchestrator.json", {
            "status": status, "completed_seeds": [seed for seed, done, _ in states if done],
            "canceled_seeds": request["canceled_seeds"], "updated": now(),
        })
        print(status, states, flush=True)
        if status != "finishing_current_runs":
            break
        time.sleep(60)


if __name__ == "__main__":
    main()

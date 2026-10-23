"""Run matched pi0 sparse/Robometer SAC arms concurrently after a baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pi0_stove.launch import build_configs
from experiments.pi0_stove.report import generate
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--suite-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pi0-python", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = root / "baseline_seed100"
    summary = json.loads((baseline / "summary.json").read_text())
    status = json.loads((baseline / "status.json").read_text())
    if status["status"] != "complete" or len(summary["records"]) != 25:
        parser.error("The 25-trial pretrained pi0 baseline must finish first")
    plans = build_configs(args.checkpoint.resolve(), args.pi0_python.absolute(), args.suite_config)
    if {plan["reward"] for plan in plans} != {"sparse", "robometer"}:
        parser.error("Suite must contain one sparse and one Robometer arm")
    if len({(plan["task"], plan["task_id"]) for plan in plans}) != 1:
        parser.error("Reward arms must use the same task")
    for plan in plans:
        plan["policy_gpu"] = 0 if plan["reward"] == "sparse" else 1
        plan["reward_gpu"] = 1
    task = plans[0]["task"]
    if summary["records"][0]["init_state_id"] != plans[0]["train_init_states"]:
        parser.error("Baseline evaluation split does not match reward arms")
    run_dir = root / f"dsrl_{task}_seed100"
    if run_dir.exists():
        parser.error(f"Refusing to overwrite {run_dir}")

    while True:
        try:
            reservation = reserve_gpus([0, 1])
            reservation.__enter__()
            break
        except RuntimeError as error:
            print(f"Waiting for idle GPUs 0 and 1: {error}", flush=True)
            time.sleep(30)
    try:
        run_dir.mkdir(exist_ok=False)
        write_json(run_dir / "planned_runs.json", {"runs": plans, "baseline": str(baseline)})
        processes = {}
        for plan in plans:
            kind = plan["reward"]
            gpu = plan["policy_gpu"]
            folder = run_dir / f"{kind}_seed100"
            folder.mkdir()
            write_json(folder / "config.json", plan)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
                   "MUJOCO_EGL_DEVICE_ID": str(gpu), "EGL_DEVICE_ID": str(gpu),
                   "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
                   "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            log = (folder / "console.log").open("w")
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                     "--config", str(folder / "config.json"), "--out", str(folder)],
                    cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            finally:
                log.close()
            processes[kind] = process

        while True:
            states = {}
            for kind, process in processes.items():
                state_path = run_dir / f"{kind}_seed100/status.json"
                states[kind] = json.loads(state_path.read_text()) if state_path.exists() else {"status": "starting"}
            write_json(run_dir / "execution.json", {
                "status": "running" if any(p.poll() is None for p in processes.values()) else "finished",
                "updated": datetime.now(timezone.utc).isoformat(),
                "pids": {kind: p.pid for kind, p in processes.items()},
                "returncodes": {kind: p.poll() for kind, p in processes.items()},
                "run_status": states,
                "gpu_assignments": {"sparse_policy": 0, "robometer_policy": 1, "robometer_reward": 1},
            })
            generate(run_dir)
            if all(p.poll() is not None for p in processes.values()):
                break
            time.sleep(30)
        if any(p.returncode != 0 or states[kind]["status"] != "complete"
               for kind, p in processes.items()):
            raise RuntimeError(f"One or both arms failed: {states}")
        write_json(run_dir / "execution.json", {
            "status": "complete", "updated": datetime.now(timezone.utc).isoformat(),
            "pids": {kind: p.pid for kind, p in processes.items()}, "run_status": states,
        })
        generate(run_dir)
    finally:
        reservation.__exit__(None, None, None)


if __name__ == "__main__":
    main()

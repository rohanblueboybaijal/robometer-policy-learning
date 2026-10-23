"""Adopt a live sparse pi0 run and launch the paired Robometer arm on GPU 1."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pi0_stove.launch import build_configs
from experiments.pi0_stove.report import generate
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


def expected_process(pid, fragment):
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    return fragment in command


def run(root, checkpoint, pi0_python, sparse_pid, old_coordinator_pid):
    output = root / "dsrl_stove_seed100"
    sparse = output / "sparse_seed100"
    if not expected_process(sparse_pid, "experiments.pi0_stove.rl"):
        raise RuntimeError(f"Sparse trainer PID {sparse_pid} is not live")
    if json.loads((sparse / "status.json").read_text())["status"] != "running":
        raise RuntimeError("Sparse trainer is not running")
    if not expected_process(old_coordinator_pid, "experiments.pi0_stove.launch"):
        raise RuntimeError(f"Old coordinator PID {old_coordinator_pid} is not live")
    baseline = json.loads((root / "baseline_seed100/summary.json").read_text())
    if baseline["successes"] != 25 or baseline["episodes"] != 25:
        raise RuntimeError("The 25-trial pi0 baseline is incomplete")
    cfg = next(c for c in build_configs(checkpoint, pi0_python) if c["reward"] == "robometer")
    cfg["policy_gpu"] = 1
    cfg["reward_gpu"] = 1
    folder = output / "robometer_seed100"
    process = None

    def save(status, **extra):
        write_json(output / "execution.json", {
            "status": status,
            "pid": os.getpid(),
            "updated": datetime.now(timezone.utc).isoformat(),
            "sparse_pid": sparse_pid,
            "robometer_pid": process.pid if process is not None else None,
            "gpu_assignments": {"sparse_policy": 0, "robometer_policy": 1, "robometer_reward": 1},
            **extra,
        })

    with reserve_gpus([1]):
        folder.mkdir(exist_ok=False)
        write_json(folder / "config.json", cfg)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "MUJOCO_GL": "egl",
               "MUJOCO_EGL_DEVICE_ID": "1", "EGL_DEVICE_ID": "1",
               "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
               "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        with (folder / "console.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                 "--config", str(folder / "config.json"), "--out", str(folder)],
                cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            save("running", old_coordinator_pid=old_coordinator_pid)
            # The existing sparse child has its own session; stopping its old
            # coordinator releases only the GPU-0 lock, not the trainer.
            if expected_process(old_coordinator_pid, "experiments.pi0_stove.launch"):
                os.kill(old_coordinator_pid, signal.SIGTERM)
            while process.poll() is None:
                if not expected_process(sparse_pid, "experiments.pi0_stove.rl"):
                    sparse_status = json.loads((sparse / "status.json").read_text())["status"]
                else:
                    sparse_status = "running"
                save("running", sparse_status=sparse_status)
                generate(output)
                time.sleep(30)
        state_path = folder / "status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "missing"}
        if process.returncode or state["status"] != "complete":
            save("robometer_failed", run_status=state)
            generate(output)
            raise RuntimeError(f"Robometer arm failed: {state}")
        save("robometer_complete")
        generate(output)
    while expected_process(sparse_pid, "experiments.pi0_stove.rl"):
        save("waiting_for_sparse")
        generate(output)
        time.sleep(30)
    sparse_state = json.loads((sparse / "status.json").read_text())
    save("complete" if sparse_state["status"] == "complete" else "sparse_failed",
         sparse_status=sparse_state)
    generate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pi0-python", required=True, type=Path)
    parser.add_argument("--sparse-pid", required=True, type=int)
    parser.add_argument("--old-coordinator-pid", required=True, type=int)
    args = parser.parse_args()
    run(args.root.resolve(), args.checkpoint.resolve(), args.pi0_python.absolute(),
        args.sparse_pid, args.old_coordinator_pid)


if __name__ == "__main__":
    main()

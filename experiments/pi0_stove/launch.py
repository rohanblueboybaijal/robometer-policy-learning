"""Wait for baseline, then run matched pi0 steering arms without sharing GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pi0_stove.report import generate
from experiments.reward_audit.config import HERE, REPO, load_suite, materialize, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def build_configs(checkpoint, pi0_python, suite_path=None):
    suite = load_suite(suite_path or HERE / "configs/stove.json")
    plans = []
    for base in suite["runs"]:
        cfg = materialize(suite, base)
        cfg.update(seed=100, num_envs=1, pi0_checkpoint=str(checkpoint), pi0_python=str(pi0_python),
                   policy_gpu=0, reward_gpu=1,
                   dsrl={"method": "SAC additive Gaussian noise steering", "noise_dim": 32, "action_exec_len": 20},
                   steps=100000, eval_every=5000, save_every=10000, log_every=1000, eval_videos=0,
                   reward_python=str((REPO / ".venv-reward/bin/python").absolute()),
                   reward_formula="progress_chunk_mean" if cfg["reward"] == "robometer" else "sparse_chunk_terminal")
        cfg["sac"].update(learning_starts=250, gamma=0.999 ** 20, compute_chunked_gamma=False,
                          num_critic_updates_per_actor_update=25, num_updates_per_train_step=1,
                          actor_optimizer_lr=1e-4, critic_optimizer_lr=3e-4, target_entropy=-16,
                          clip_grad_norm=10.0)
        plans.append(cfg)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi0-python", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = root / "baseline_seed100"
    checkpoint = args.checkpoint.resolve()
    pi0_python = args.pi0_python.absolute()  # Keep venv symlink for site-packages.
    if not (checkpoint / "assets").exists() or not (checkpoint / "params").exists():
        parser.error("Pi0 checkpoint is incomplete")
    if not pi0_python.exists():
        parser.error("Pi0 Python is missing")
    plans = build_configs(checkpoint, pi0_python)
    output = root / "dsrl_stove_seed100"
    output.mkdir(exist_ok=False)
    write_json(output / "planned_runs.json", {"runs": plans, "baseline": str(baseline)})

    def save(status, **details):
        write_json(output / "execution.json", {"status": status, "pid": os.getpid(),
                                               "updated": datetime.now(timezone.utc).isoformat(), **details})

    baseline_pid = json.loads((root / "baseline_process.json").read_text())["pid"]
    while True:
        status_path = baseline / "status.json"
        if status_path.exists():
            state = json.loads(status_path.read_text())
            if state["status"] == "complete":
                if len(json.loads((baseline / "summary.json").read_text())["records"]) != 25:
                    raise RuntimeError("Baseline lacks 25 evaluations")
                break
            if state["status"] == "failed":
                raise RuntimeError("Pi0 baseline failed; refusing reward comparison")
        if not alive(baseline_pid):
            raise RuntimeError("Pi0 baseline ended without a complete result")
        save("waiting_for_baseline", baseline_pid=baseline_pid)
        time.sleep(20)

    for cfg in plans:
        kind = cfg["reward"]
        indices = [0] if kind == "sparse" else [0, 1]
        while True:
            try:
                reservation = reserve_gpus(indices)
                reservation.__enter__()
                break
            except RuntimeError as error:
                save("waiting_for_gpus", arm=kind, reason=str(error))
                time.sleep(30)
        try:
            folder = output / f"{kind}_seed100"
            folder.mkdir(exist_ok=False)
            write_json(folder / "config.json", cfg)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl",
                   "MUJOCO_EGL_DEVICE_ID": "0", "EGL_DEVICE_ID": "0",
                   "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
                   "WANDB_MODE": "disabled", "WANDB_DISABLED": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            with (folder / "console.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                                            "--config", str(folder / "config.json"), "--out", str(folder)],
                                           cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    save("running", arm=kind, run_pid=process.pid, gpu_assignments={"policy": 0,
                                                                                   "reward": 1 if kind == "robometer" else None})
                    generate(output)
                    time.sleep(30)
            state_path = folder / "status.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "missing"}
            if process.returncode != 0 or state["status"] != "complete":
                raise RuntimeError(f"{kind} run failed: {state}")
            save("arm_complete", arm=kind)
            generate(output)
        finally:
            reservation.__exit__(None, None, None)
    save("complete")
    generate(output)


if __name__ == "__main__":
    main()

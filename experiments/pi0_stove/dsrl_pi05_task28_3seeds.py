"""Three paired pi0.5 direct-DSRL seeds on LIBERO-90 task 28."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.pi0_stove.dsrl_pi05_libero90 import CHECKPOINT, DATA_ROOT, config as prior_config
from experiments.pi0_stove.dsrl_smoke import CACHE
from experiments.pi0_stove.dsrl_multitask import now
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

ROOT = DATA_ROOT / "pi05_libero90_task28_dsrl_3seeds_20261022"
SEEDS = (100, 101, 102)
KINDS = ("sparse", "robometer")
STEPS = 100_000
REWARD_CHECKPOINT = "robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34"


def run_dir(seed: int, kind: str) -> Path:
    return ROOT / f"seed{seed}" / f"{kind}_seed{seed}"


def config(seed: int, kind: str, gpu: int) -> dict:
    cfg = prior_config(28, seed, kind, gpu)
    cfg.update(
        name=f"pi05_libero90_task28_dsrl_{kind}",
        comparison="pi05_libero90_task28_direct_dsrl_3seeds_20261022",
        reward_checkpoint=REWARD_CHECKPOINT,
        eval_episodes=25,
        eval_unique_states=25,
        eval_every=10_000,
        rolling_checkpoint_only=True,
    )
    assert cfg["pi0_checkpoint"] == str(CHECKPOINT)
    assert cfg["policy_config_name"] == "pi05_libero"
    assert cfg["dsrl"]["noise_mode"] == "direct"
    assert cfg["sac"]["learning_starts"] == 500
    assert cfg["sac"]["num_critic_updates_per_actor_update"] == 25
    return cfg


def complete(seed: int, kind: str) -> bool:
    folder = run_dir(seed, kind)
    try:
        status = json.loads((folder / "status.json").read_text())
    except (OSError, ValueError):
        return False
    return (status.get("status") == "complete" and status.get("env_steps") == STEPS
            and (folder / "checkpoints/final.pt").is_file())


def initialize() -> None:
    if not (CHECKPOINT / "params").is_dir() or not (CHECKPOINT / "assets").is_dir():
        raise RuntimeError(f"Missing pi0.5 checkpoint: {CHECKPOINT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": "libero_90", "task_id": 28,
        "task": config(100, "sparse", 0)["task"],
        "seeds": list(SEEDS), "rewards": list(KINDS),
        "steps_per_run": STEPS, "algorithm": "direct DSRL",
        "pi0_checkpoint": str(CHECKPOINT), "policy_config_name": "pi05_libero",
        "reward_checkpoint": REWARD_CHECKPOINT,
        "eval_episodes": 25, "eval_unique_states": 25,
        "eval_every": 10_000, "horizon": 400,
        "baseline_source": "seed100/sparse_seed100/evaluation.csv at 0 transitions",
    }
    path = ROOT / "manifest.json"
    if path.exists():
        current = json.loads(path.read_text())
        if any(current.get(key) != value for key, value in manifest.items()):
            raise RuntimeError(f"Unexpected manifest: {path}")
    else:
        write_json(path, {**manifest, "created": now()})


def launch(seed: int, kind: str, gpu: int):
    folder = run_dir(seed, kind)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", config(seed, kind, gpu))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu),
           "EGL_DEVICE_ID": str(gpu), "AUDIT_LAUNCHER_GUARDED": "1",
           "PYTHONPATH": source_pythonpath(), "WANDB_MODE": "disabled",
           "WANDB_DISABLED": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XDG_CACHE_HOME": str(CACHE), "HF_HOME": str(CACHE / "hf"),
           "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "torchinductor"),
           "TRITON_CACHE_DIR": str(CACHE / "triton"),
           "TMPDIR": "/data/rohan/rbm-tmp"}
    log = (folder / "console.log").open("w")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
             "--config", str(folder / "config.json"), "--out", str(folder)],
            cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    return process, log


def status(active, pending, errors):
    write_json(ROOT / "orchestrator.json", {
        "status": "failed" if errors else ("running" if active or pending else "complete"),
        "pid": os.getpid(), "updated": now(), "errors": errors,
        "active": {str(gpu): {"seed": seed, "kind": kind, "pid": process.pid,
                              "run": str(run_dir(seed, kind))}
                   for gpu, (seed, kind, process, _, _) in active.items()},
        "pending": [{"seed": seed, "kind": kind} for seed, kind in pending],
    })


def run() -> None:
    initialize()
    jobs = [(seed, kind) for seed in SEEDS for kind in KINDS]
    pending = [job for job in jobs if not complete(*job)]
    for job in pending:
        if run_dir(*job).exists():
            raise RuntimeError(f"Incomplete run needs inspection: {run_dir(*job)}")
    active = {}
    errors = []
    while pending or active:
        for gpu, (seed, kind, process, log, reservation) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            reservation.__exit__(None, None, None)
            del active[gpu]
            if code or not complete(seed, kind):
                errors.append(f"{kind} seed {seed} exited {code}: {run_dir(seed, kind)}")
            else:
                print(f"Completed {kind} seed {seed} on GPU {gpu}", flush=True)
        if not errors:
            for gpu in (0, 1):
                if gpu in active or not pending:
                    continue
                if shutil.disk_usage(DATA_ROOT).free < 30 * 1024**3:
                    print("Waiting for 30 GiB free on /data", flush=True)
                    break
                try:
                    reservation = reserve_gpus([gpu])
                    reservation.__enter__()
                except RuntimeError:
                    continue
                seed, kind = pending.pop(0)
                try:
                    process, log = launch(seed, kind, gpu)
                except BaseException:
                    reservation.__exit__(*sys.exc_info())
                    raise
                active[gpu] = (seed, kind, process, log, reservation)
                print(f"Started {kind} seed {seed} on GPU {gpu}: PID {process.pid}", flush=True)
        status(active, pending, errors)
        if errors and not active:
            raise RuntimeError("; ".join(errors))
        time.sleep(30)
    assert all(complete(*job) for job in jobs)


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with (DATA_ROOT / "pi05_libero90_task28_dsrl_3seeds_20261022.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


if __name__ == "__main__":
    main()

"""Five paired direct-DSRL seeds on LIBERO-10 task 06, across two idle GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.pi0_stove.dsrl_multitask import now
from experiments.pi0_stove.dsrl_smoke import CACHE, config as smoke_config
from experiments.reward_audit.config import REPO, task_map, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


ROOT = Path("/data/rohan/Reward-Modeling/pi0_libero10_task06_dsrl_5seeds_20261015")
SEEDS = (100, 101, 102, 103, 104)
KINDS = ("sparse", "robometer")
STEPS = 100_000
TASK_ID = 6


def run_dir(seed: int, kind: str, root: Path = ROOT) -> Path:
    return root / f"seed{seed}" / f"{kind}_seed{seed}"


def config(seed: int, kind: str, gpu: int, task_id: int = TASK_ID) -> dict:
    cfg = smoke_config(kind, gpu)
    cfg.update(suite="libero_10", task_id=task_id,
               task=task_map()["libero_10"][task_id],
               reward=kind, seed=seed, name=f"libero10_task{task_id:02d}_dsrl_{kind}",
               comparison=("libero10_task06_direct_dsrl_5seeds" if task_id == 6 else
                           f"libero10_task{task_id:02d}_direct_dsrl_8seeds"),
               ood_claim="in_distribution", steps=STEPS, eval_every=5000,
               eval_episodes=25, train_init_states=25, save_every=10000,
               rolling_checkpoint_only=True, policy_gpu=gpu, reward_gpu=gpu)
    assert cfg["num_envs"] == 1
    assert cfg["dsrl"]["noise_mode"] == "direct"
    assert cfg["sac"]["learning_starts"] == 500
    assert cfg["sac"]["num_critic_updates_per_actor_update"] == 25
    assert cfg["sac"]["num_updates_per_train_step"] == 1
    return cfg


def complete(seed: int, kind: str, root: Path = ROOT) -> bool:
    folder = run_dir(seed, kind, root)
    status = folder / "status.json"
    if not status.exists():
        return False
    state = json.loads(status.read_text())
    return (state.get("status") == "complete" and state.get("env_steps") == STEPS and
            (folder / "checkpoints/final.pt").is_file())


def initialize(seeds: tuple[int, ...] = SEEDS, root: Path = ROOT,
               task_id: int = TASK_ID) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected = {"suite": "libero_10", "task_id": task_id,
                "task": task_map()["libero_10"][task_id],
                "seeds": list(seeds), "rewards": list(KINDS),
                "steps_per_run": STEPS, "algorithm": "direct DSRL"}
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text())
        if any(current.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected experiment manifest: {manifest_path}")
    else:
        cfg = config(seeds[0], "robometer", 0, task_id)
        write_json(manifest_path, {**expected, "created": now(),
                                   "pi0_checkpoint": cfg["pi0_checkpoint"],
                                   "reward_checkpoint": cfg["reward_checkpoint"],
                                   "baseline": "/data/rohan/Reward-Modeling/"
                                   "pi0_libero_families_20261014/libero_10"})


def launch(seed: int, kind: str, gpu: int, root: Path = ROOT,
           task_id: int = TASK_ID):
    folder = run_dir(seed, kind, root)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = config(seed, kind, gpu, task_id)
    write_json(folder / "config.json", cfg)
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


def run(seeds: tuple[int, ...] = SEEDS, root: Path = ROOT,
        task_id: int = TASK_ID) -> None:
    initialize(seeds, root, task_id)
    pending = [(seed, kind) for seed in seeds for kind in KINDS if not complete(seed, kind, root)]
    for seed, kind in pending:
        if run_dir(seed, kind, root).exists():
            raise RuntimeError(f"Incomplete run needs inspection before restart: {run_dir(seed, kind, root)}")
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
            if code or not complete(seed, kind, root):
                errors.append(f"{kind} seed {seed} failed (exit {code}); inspect {run_dir(seed, kind, root)}")
            else:
                print(f"Completed {kind} seed {seed} on GPU {gpu}", flush=True)
        if not errors:
            for gpu in (0, 1):
                if gpu in active or not pending:
                    continue
                if shutil.disk_usage(root).free < 30 * 1024**3:
                    print("Waiting for at least 30 GiB free on /data", flush=True)
                    break
                try:
                    reservation = reserve_gpus([gpu])
                    reservation.__enter__()
                except RuntimeError:
                    continue
                seed, kind = pending.pop(0)
                try:
                    process, log = launch(seed, kind, gpu, root, task_id)
                except BaseException:
                    reservation.__exit__(*sys.exc_info())
                    raise
                active[gpu] = (seed, kind, process, log, reservation)
                print(f"Started {kind} seed {seed} on GPU {gpu}: PID {process.pid}", flush=True)
        write_json(root / "orchestrator.json", {
            "status": "failed" if errors else "running",
            "pid": os.getpid(), "updated": now(), "errors": errors,
            "active": {str(gpu): {"seed": seed, "kind": kind, "pid": process.pid,
                                  "run": str(run_dir(seed, kind, root))}
                       for gpu, (seed, kind, process, _, _) in active.items()},
            "pending": [{"seed": seed, "kind": kind} for seed, kind in pending],
        })
        if errors and not active:
            raise RuntimeError("; ".join(errors))
        time.sleep(30)
    assert all(complete(seed, kind, root) for seed in seeds for kind in KINDS)
    write_json(root / "orchestrator.json", {"status": "complete", "pid": os.getpid(),
                                            "updated": now(), "active": {}, "pending": []})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Coordinator already running: {ROOT}") from None
        run()


if __name__ == "__main__":
    main()

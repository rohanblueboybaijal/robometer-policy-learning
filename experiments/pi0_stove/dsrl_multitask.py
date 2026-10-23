"""Run three paired direct-DSRL seeds on task 38, then task 55."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pi0_stove.dsrl_smoke import CACHE, config as smoke_config
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


TASKS = {
    38: "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
    55: "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
}
SEEDS = (100, 101, 102)
KINDS = ("sparse", "robometer")
STEPS = 100000
DATA_ROOT = Path("/data/rohan/Reward-Modeling")
ROOT_NAME = "dsrl_task38_then55_20261012"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stage_root(task_id: int) -> Path:
    return DATA_ROOT / f"pi0_libero90_task{task_id}_dsrl_3seeds_20261012"


def run_dir(task_id: int, kind: str, seed: int) -> Path:
    return stage_root(task_id) / f"seed{seed}" / f"{kind}_seed{seed}"


def plan_config(task_id: int, kind: str, seed: int, gpu: int) -> dict:
    cfg = smoke_config(kind, gpu)
    cfg.update(task=TASKS[task_id], task_id=task_id, reward=kind, seed=seed,
               name=f"task{task_id}_dsrl_{kind}", comparison=f"libero90_task{task_id}_direct_dsrl",
               steps=STEPS, eval_every=5000, eval_episodes=25, save_every=10000,
               policy_gpu=gpu, reward_gpu=gpu)
    assert cfg["num_envs"] == 1
    assert cfg["sac"]["learning_starts"] == 500
    assert cfg["sac"]["num_critic_updates_per_actor_update"] == 25
    assert cfg["dsrl"]["noise_mode"] == "direct"
    return cfg


def initialize(root: Path, resume: bool) -> None:
    if resume:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("tasks") != [38, 55] or manifest.get("seeds") != list(SEEDS):
            raise RuntimeError("Unexpected DSRL experiment manifest")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "manifest.json", {"tasks": [38, 55], "seeds": SEEDS,
                   "rewards": KINDS, "steps_per_run": STEPS,
                   "algorithm": "direct DSRL", "created": now()})
    for task_id in TASKS:
        stage = stage_root(task_id)
        if stage.exists():
            manifest = json.loads((stage / "manifest.json").read_text())
            if manifest.get("task_id") != task_id or manifest.get("seeds") != list(SEEDS):
                raise RuntimeError(f"Unexpected stage manifest: {stage}")
        else:
            stage.mkdir(parents=True)
            write_json(stage / "manifest.json", {"task_id": task_id, "task": TASKS[task_id],
                       "seeds": SEEDS, "steps_per_run": STEPS,
                       "algorithm": "direct DSRL", "source": str(root), "created": now()})


def complete(run: Path) -> bool:
    status_file = run / "status.json"
    if not status_file.exists():
        return False
    state = json.loads(status_file.read_text())
    return (state.get("status") == "complete" and state.get("env_steps") == STEPS
            and (run / "checkpoints/final.pt").exists())


def process_alive(pid: int, run: Path) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return False
    command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return "experiments.pi0_stove.rl" in command and str(run) in command


def wait_for_orphans(root: Path) -> None:
    """On resume, let detached training jobs finish before scheduling new work."""
    state_file = root / "orchestrator.json"
    if not state_file.exists():
        return
    previous = json.loads(state_file.read_text())
    active = previous.get("active", {})
    for details in active.values():
        run = Path(details["run"])
        pid = int(details["pid"])
        while process_alive(pid, run):
            write_json(state_file, {"status": "waiting_for_existing_run", "active": active,
                                    "updated": now()})
            time.sleep(30)


def launch_run(task_id: int, kind: str, seed: int, gpu: int) -> tuple[subprocess.Popen, object]:
    run = run_dir(task_id, kind, seed)
    if run.exists():
        if complete(run):
            raise RuntimeError(f"Already complete: {run}")
        state_file = run / "status.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get("status") not in ("failed", "running"):
            raise RuntimeError(f"Refusing to overwrite incomplete run {run}: {state}")
        attempt = 1
        while run.with_name(f"{run.name}.failed_attempt{attempt}").exists():
            attempt += 1
        run.rename(run.with_name(f"{run.name}.failed_attempt{attempt}"))
    run.mkdir(parents=True)
    write_json(run / "config.json", plan_config(task_id, kind, seed, gpu))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
           "MUJOCO_EGL_DEVICE_ID": str(gpu), "EGL_DEVICE_ID": str(gpu),
           "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
           "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XDG_CACHE_HOME": str(CACHE),
           "HF_HOME": str(CACHE / "hf"), "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "torchinductor"),
           "TRITON_CACHE_DIR": str(CACHE / "triton"), "TMPDIR": "/data/rohan/rbm-tmp"}
    log = (run / "console.log").open("w")
    process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                                "--config", str(run / "config.json"), "--out", str(run)],
                               cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def stage(root: Path, task_id: int, seeds: tuple[int, ...] = SEEDS) -> None:
    # Interleave reward arms so either available GPU makes paired-seed progress.
    pending = [(kind, seed) for seed in seeds for kind in ("robometer", "sparse")
               if not complete(run_dir(task_id, kind, seed))]
    retries = {(kind, seed): 0 for kind, seed in pending}
    active = {}
    while pending or active:
        for gpu, (kind, seed, process, log, reservation) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            reservation.__exit__(None, None, None)
            del active[gpu]
            run = run_dir(task_id, kind, seed)
            if code == 0 and complete(run):
                print(f"Completed task {task_id}, {kind}, seed {seed} on GPU {gpu}", flush=True)
            elif retries[(kind, seed)] < 2:
                retries[(kind, seed)] += 1
                pending.insert(0, (kind, seed))
                print(f"Retrying task {task_id}, {kind}, seed {seed}: exit {code}", flush=True)
            else:
                raise RuntimeError(f"Run failed after retries: {run}; see console.log")
        for gpu in (0, 1):
            if gpu in active or not pending:
                continue
            if shutil.disk_usage(DATA_ROOT).free < 30 * 1024**3:
                print("Waiting for at least 30 GiB free on /data", flush=True)
                break
            try:
                reservation = reserve_gpus([gpu])
                reservation.__enter__()
            except RuntimeError:
                continue  # This GPU belongs to another job; check again later.
            kind, seed = pending.pop(0)
            try:
                process, log = launch_run(task_id, kind, seed, gpu)
            except BaseException:
                reservation.__exit__(*sys.exc_info())
                raise
            active[gpu] = (kind, seed, process, log, reservation)
            print(f"Started task {task_id}, {kind}, seed {seed} on GPU {gpu}: PID {process.pid}", flush=True)
        write_json(root / "orchestrator.json", {
            "status": f"running_task{task_id}", "pid": os.getpid(), "task_id": task_id,
            "active": {str(gpu): {"kind": kind, "seed": seed, "pid": process.pid,
                                  "run": str(run_dir(task_id, kind, seed))}
                       for gpu, (kind, seed, process, _, _) in active.items()},
            "pending": [{"kind": kind, "seed": seed} for kind, seed in pending],
            "updated": now(),
        })
        time.sleep(30)
    if not all(complete(run_dir(task_id, kind, seed)) for kind in KINDS for seed in seeds):
        raise RuntimeError(f"Task {task_id} stage ended with incomplete runs")
    write_json(root / "orchestrator.json", {"status": f"task{task_id}_complete",
                                                "pid": os.getpid(), "updated": now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT / ROOT_NAME)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    initialize(root, args.resume)
    if args.resume:
        wait_for_orphans(root)
    for task_id in (38, 55):
        stage(root, task_id)
    write_json(root / "orchestrator.json", {"status": "complete", "pid": os.getpid(), "updated": now()})


if __name__ == "__main__":
    main()

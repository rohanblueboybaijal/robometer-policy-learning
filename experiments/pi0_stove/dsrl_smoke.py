"""Short, paired task-55 DSRL smoke test; never starts the full runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


SOURCE = Path("/data/rohan/Reward-Modeling/pi0_libero90_task55_20261010") / (
    "dsrl_LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_seed100"
)
CACHE = Path("/data/rohan/Reward-Modeling/pi0_libero90_task38_five_seeds_20261011/cache")


def config(kind: str, gpu: int) -> dict:
    cfg = json.loads((SOURCE / f"{kind}_seed100" / "config.json").read_text())
    cfg.update(reward=kind, seed=100, policy_gpu=gpu, reward_gpu=gpu,
               comparison="libero90_task55_dsrl_smoke", name=f"task55_dsrl_smoke_{kind}",
               steps=700, eval_every=700, eval_episodes=2, save_every=100,
               eval_videos=0)
    cfg["dsrl"] = {"method": "SAC direct pi0 noise (original DSRL semantics)",
                   "noise_dim": 32, "action_exec_len": 20, "noise_mode": "direct"}
    cfg["sac"]["learning_starts"] = 500  # Original LIBERO DSRL: physical transitions.
    assert cfg["sac"]["num_critic_updates_per_actor_update"] == 25
    assert cfg["sac"]["num_updates_per_train_step"] == 1
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.retry_failed:
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest.get("smoke_test") or manifest.get("task_id") != 55:
            raise RuntimeError(f"Refusing to retry a non-smoke run at {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "manifest.json", {"task_id": 55,
                   "task": "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
                   "seeds": [100], "steps_per_run": 700, "smoke_test": True,
                   "full_runs_started": False})
    processes = {}
    planned = []
    for kind, gpu in (("sparse", 0), ("robometer", 1)):
        run = root / "seed100" / f"{kind}_seed100"
        if run.exists():
            state_file = run / "status.json"
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            if state.get("status") == "complete" and (run / "checkpoints/final.pt").exists():
                print(f"Skipping completed {kind} smoke test", flush=True)
                continue
            if not args.retry_failed or state.get("status") != "failed":
                raise RuntimeError(f"Refusing to overwrite {run}: {state}")
            attempt = 1
            while run.with_name(f"{run.name}.failed_attempt{attempt}").exists():
                attempt += 1
            run.rename(run.with_name(f"{run.name}.failed_attempt{attempt}"))
        planned.append((kind, gpu))
    with reserve_gpus([gpu for _, gpu in planned]):
        for kind, gpu in planned:
            run = root / "seed100" / f"{kind}_seed100"
            run.mkdir(parents=True)
            write_json(run / "config.json", config(kind, gpu))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu),
                   "EGL_DEVICE_ID": str(gpu), "AUDIT_LAUNCHER_GUARDED": "1",
                   "PYTHONPATH": source_pythonpath(), "WANDB_MODE": "disabled",
                   "WANDB_DISABLED": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XDG_CACHE_HOME": str(CACHE), "HF_HOME": str(CACHE / "hf"),
                   "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "torchinductor"),
                   "TRITON_CACHE_DIR": str(CACHE / "triton"),
                   "TMPDIR": "/data/rohan/rbm-tmp"}
            log = (run / "console.log").open("w")
            processes[kind] = (subprocess.Popen(
                [sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                 "--config", str(run / "config.json"), "--out", str(run)],
                cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log)
            print(f"Started {kind} smoke test on GPU {gpu}: PID {processes[kind][0].pid}", flush=True)
        returncodes = {}
        for kind, (process, log) in processes.items():
            returncodes[kind] = process.wait()
            log.close()
            print(f"{kind} exit code: {returncodes[kind]}", flush=True)
    results = {}
    for kind in ("sparse", "robometer"):
        run = root / "seed100" / f"{kind}_seed100"
        status = json.loads((run / "status.json").read_text())
        checkpoints = sorted(path.name for path in (run / "checkpoints").glob("*.pt"))
        results[kind] = {"status": status.get("status"), "env_steps": status.get("env_steps"),
                         "checkpoints": checkpoints}
    write_json(root / "smoke_status.json", {"returncodes": returncodes, "results": results,
                                               "full_runs_started": False})
    if any(returncodes.values()) or any(result["status"] != "complete" or
                                       result["env_steps"] != 700 or
                                       result["checkpoints"] != ["final.pt", "latest.pt"]
                                       for result in results.values()):
        raise RuntimeError(f"One or more task-55 DSRL smoke tests failed: {results}")


if __name__ == "__main__":
    main()

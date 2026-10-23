"""Run and report five matched pi0 SAC seeds on LIBERO-90 task 38."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pi0_stove.multiseed import aggregate
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

TASK = "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove"
SEEDS = (100, 101, 102, 103, 104)
LANES = {
    0: (("sparse", 100), ("robometer", 100), ("sparse", 102), ("robometer", 102), ("robometer", 104)),
    1: (("sparse", 101), ("robometer", 101), ("sparse", 103), ("robometer", 103), ("sparse", 104)),
}
SOURCE = Path("/data/rohan/Reward-Modeling/pi0_libero90_task55_20261010") / (
    "dsrl_LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_seed100"
)


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(root, kind, seed):
    return root / f"seed{seed}" / f"{kind}_seed{seed}"


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stop_requested = (root / "stop_after_seed101.json").exists()
    curves = {}
    for kind in ("sparse", "robometer"):
        curves[kind] = {}
        for seed in SEEDS:
            path = folder(root, kind, seed) / "evaluation.csv"
            try:
                with path.open() as stream:
                    entries = []
                    for row in csv.DictReader(stream):
                        try:
                            entries.append({"env_steps": int(float(row["env_steps"])),
                                            "successes": int(row["successes"]),
                                            "episodes": int(row["episodes"]),
                                            "success_rate": float(row["success_rate"])})
                        except (TypeError, ValueError):
                            continue
                    curves[kind][seed] = entries
            except FileNotFoundError:
                curves[kind][seed] = []
    means = {kind: aggregate(curves[kind]) for kind in curves}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for kind, color in (("sparse", "#087f8c"), ("robometer", "#d36a35")):
        for entries in curves[kind].values():
            if entries:
                ax.plot([r["env_steps"] for r in entries],
                        [r["success_rate"] for r in entries], color=color, alpha=.22)
        points = means[kind]
        if points:
            xs = [p["env_steps"] for p in points]
            ax.plot(xs, [p["mean"] for p in points], color=color, linewidth=2.6,
                    marker="o", markersize=3, label=f"{kind.title()} mean")
            ax.fill_between(xs, [p["ci_low"] for p in points],
                            [p["ci_high"] for p in points], color=color, alpha=.11)
    ax.set(xlabel="Training environment transitions", ylabel="Simulator success rate",
           xlim=(0, 100000), ylim=(-.03, 1.03),
           title="LIBERO-90 task 38: pi0 SAC steering across two seeds" if stop_requested
           else "LIBERO-90 task 38: pi0 SAC steering across five seeds")
    ax.grid(alpha=.2)
    if any(means.values()):
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "success_rate_five_seeds.png", dpi=180)
    fig.savefig(root / "success_rate_five_seeds.pdf")
    plt.close(fig)
    write_json(root / "curves.json", {"individual": curves, "aggregate": means, "updated": now()})
    rows = []
    for seed in SEEDS:
        for kind in ("sparse", "robometer"):
            run = folder(root, kind, seed)
            status_file = run / "status.json"
            status = json.loads(status_file.read_text()).get("status", "unknown") if status_file.exists() else "queued"
            if stop_requested and seed > 101 and status == "queued":
                status = "canceled"
            last = curves[kind][seed][-1] if curves[kind][seed] else None
            rows.append(f"<tr><td>{seed}</td><td>{kind}</td><td>{status}</td>"
                        f"<td>{last['env_steps']:,}</td><td>{last['successes']}/{last['episodes']}</td></tr>"
                        if last else f"<tr><td>{seed}</td><td>{kind}</td><td>{status}</td><td>—</td><td>—</td></tr>")
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='120'>"
        "<title>π₀ LIBERO-90 task 38</title><style>body{font:16px/1.5 system-ui;max-width:1050px;"
        "margin:auto;padding:30px;color:#17313b;background:#f5f7f6}main{background:white;"
        "border-radius:14px;padding:28px}img{width:100%}table{border-collapse:collapse;width:100%}"
        "td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style></head><body><main>"
        "<h1>π₀ on LIBERO-90 task 38</h1><p>Put the right moka pot on the stove. "
        + ("Two matched seeds were run; seeds 102–104 were canceled. " if stop_requested
           else "Five matched seeds were scheduled. ")
        + "The runs compare sparse simulator success reward with Robometer progress reward. "
        "The π₀ checkpoint is frozen; SAC trains residual action steering. "
        "Each evaluation uses 25 held-out initial states. Shaded bands are 95% t intervals "
        "over available seeds, so sample size varies until all runs finish.</p>"
        "<img src='success_rate_five_seeds.png' alt='Success rate versus transitions'>"
        "<p><a href='success_rate_five_seeds.pdf'>PDF</a> · <a href='curves.json'>Curve data</a></p>"
        "<table><thead><tr><th>Seed</th><th>Reward</th><th>Status</th><th>Transitions</th>"
        "<th>Successes</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></main></body></html>"
    )


def run_lane(root, gpu):
    with reserve_gpus([gpu]):
        for kind, seed in LANES[gpu]:
            run = folder(root, kind, seed)
            if run.exists():
                status_file = run / "status.json"
                state = json.loads(status_file.read_text()) if status_file.exists() else {}
                if (state.get("status") == "complete" and state.get("env_steps") == 100000
                        and (run / "checkpoints/final.pt").exists()):
                    continue
                if state.get("status") != "failed":
                    raise RuntimeError(f"Refusing to overwrite nonfailed run {run}: {state}")
                attempt = 1
                while run.with_name(f"{run.name}.failed_attempt{attempt}").exists():
                    attempt += 1
                run.rename(run.with_name(f"{run.name}.failed_attempt{attempt}"))
            run.mkdir(parents=True)
            cfg = json.loads((SOURCE / f"{kind}_seed100" / "config.json").read_text())
            cfg.update(task=TASK, task_id=38, name=f"task38_{kind}", comparison="libero90_task38",
                       seed=seed, policy_gpu=gpu, reward_gpu=gpu)
            write_json(run / "config.json", cfg)
            cache = root / "cache"
            for directory in (cache, cache / "hf", cache / "torchinductor", cache / "triton"):
                directory.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
                   "MUJOCO_EGL_DEVICE_ID": str(gpu), "EGL_DEVICE_ID": str(gpu),
                   "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
                   "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                   "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "hf"),
                   "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                   "TRITON_CACHE_DIR": str(cache / "triton"), "TMPDIR": "/data/rohan/rbm-tmp"}
            with (run / "console.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                     "--config", str(run / "config.json"), "--out", str(run)],
                    cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                write_json(root / f"lane_{gpu}.json", {"status": "running", "gpu": gpu,
                            "kind": kind, "seed": seed, "pid": process.pid, "updated": now()})
                if process.wait() != 0:
                    raise RuntimeError(f"{kind} seed {seed} exited {process.returncode}; see {run / 'console.log'}")
            state = json.loads((run / "status.json").read_text())
            if state.get("status") != "complete" or state.get("env_steps") != 100000:
                raise RuntimeError(f"{kind} seed {seed} ended incomplete: {state}")
        write_json(root / f"lane_{gpu}.json", {"status": "complete", "gpu": gpu, "updated": now()})


def launch(root, resume=False):
    if resume:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("task_id") != 38 or manifest.get("seeds") != list(SEEDS):
            raise RuntimeError(f"Unexpected experiment manifest at {root}")
        state = json.loads((root / "orchestrator.json").read_text())
        if state.get("status") == "running":
            raise RuntimeError("Experiment orchestrator is already running")
    else:
        if root.exists():
            raise RuntimeError(f"Refusing to overwrite {root}")
        root.mkdir(parents=True)
        write_json(root / "manifest.json", {"task_id": 38, "task": TASK, "seeds": SEEDS,
                   "steps_per_run": 100000, "lanes": LANES, "source_config": str(SOURCE), "created": now()})
    processes = {}
    for gpu in (0, 1):
        with (root / f"lane_{gpu}.log").open("w") as log:
            processes[gpu] = subprocess.Popen(
                [sys.executable, "-u", "-m", "experiments.pi0_stove.task38_five_seeds",
                 "lane", "--root", str(root), "--gpu", str(gpu)],
                cwd=REPO, env=os.environ, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
    while any(process.poll() is None for process in processes.values()):
        report(root)
        write_json(root / "orchestrator.json", {"status": "running", "pid": os.getpid(),
                   "lane_pids": {str(g): p.pid for g, p in processes.items()},
                   "lane_returncodes": {str(g): p.poll() for g, p in processes.items()}, "updated": now()})
        time.sleep(60)
    report(root)
    status = "complete" if all(p.returncode == 0 for p in processes.values()) else "failed"
    write_json(root / "orchestrator.json", {"status": status, "pid": os.getpid(),
               "lane_returncodes": {str(g): p.returncode for g, p in processes.items()}, "updated": now()})
    if status != "complete":
        raise RuntimeError("One or more GPU lanes failed; inspect lane logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "resume", "lane", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in ("launch", "resume"):
        launch(root, resume=args.command == "resume")
    elif args.command == "lane":
        if args.gpu is None:
            parser.error("lane requires --gpu")
        run_lane(root, args.gpu)
    else:
        report(root)


if __name__ == "__main__":
    main()

"""Run four more paired LIBERO-90 pi0 seeds and plot the five-seed curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.gpu_guard import assert_idle
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

SEEDS = (100, 101, 102, 103, 104)
NEW_SEEDS = (101, 102, 103, 104)
KINDS = ("sparse", "robometer")
TASK = "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray"
LANES = {0: (("sparse", 101), ("sparse", 103), ("robometer", 101), ("robometer", 103)),
         1: (("sparse", 102), ("sparse", 104), ("robometer", 102), ("robometer", 104))}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.split(") ", 1)[1][0] != "Z"


def trainer_alive(pid, folder):
    if not process_alive(pid):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    return "experiments.pi0_stove.rl" in command and str(folder) in command


def source_dir(root, seed, kind):
    manifest = json.loads((root / "manifest.json").read_text())
    if seed == 100:
        return Path(manifest["seed100_run"]) / f"{kind}_seed100"
    return root / f"seed{seed}" / f"{kind}_seed{seed}"


def rows_at(path):
    if not path.exists():
        return []
    with path.open() as stream:
        rows = []
        for row in csv.DictReader(stream):
            try:
                rows.append({"env_steps": int(float(row["env_steps"])),
                             "successes": int(float(row["successes"])),
                             "episodes": int(float(row["episodes"])),
                             "success_rate": float(row["success_rate"])})
            except (TypeError, ValueError):
                continue
    return rows


def aggregate(curves):
    """Align evaluations by 5k checkpoint index; physical counts differ slightly."""
    points = []
    for index in range(21):
        values = [curves[seed][index]["success_rate"] for seed in SEEDS
                  if len(curves[seed]) > index]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
            half_width = T95[n] * sample_std / math.sqrt(n)
        else:
            half_width = 0.0
        points.append({"env_steps": min(index * 5000, 100000), "n": n, "mean": mean,
                       "ci_low": max(0.0, mean - half_width),
                       "ci_high": min(1.0, mean + half_width)})
    return points


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {kind: {seed: rows_at(source_dir(root, seed, kind) / "evaluation.csv")
                     for seed in SEEDS} for kind in KINDS}
    aggregates = {kind: aggregate(curves[kind]) for kind in KINDS}
    figure, axis = plt.subplots(figsize=(10, 5.4))
    for kind, color in (("sparse", "#087f8c"), ("robometer", "#d36a35")):
        for seed in SEEDS:
            entries = curves[kind][seed]
            if entries:
                axis.plot([row["env_steps"] for row in entries],
                          [row["success_rate"] for row in entries],
                          color=color, alpha=.22, linewidth=1)
        mean = aggregates[kind]
        if mean:
            x = [point["env_steps"] for point in mean]
            axis.plot(x, [point["mean"] for point in mean], color=color,
                      linewidth=2.6, marker="o", markersize=3,
                      label=f"{kind.title()} mean (available seeds)")
            axis.fill_between(x, [point["ci_low"] for point in mean],
                              [point["ci_high"] for point in mean], color=color, alpha=.11)
    axis.set(xlabel="Training environment transitions", ylabel="Held-out simulator success rate",
             xlim=(0, 100000), ylim=(-.03, 1.03),
             title="LIBERO-90 task 55: pi0 SAC steering across five seeds")
    axis.grid(alpha=.2)
    if any(aggregates.values()):
        axis.legend()
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(root / f"success_rate_five_seeds.{extension}", dpi=180)
    plt.close(figure)
    write_json(root / "curves.json", {"individual": curves, "aggregate": aggregates})
    table = []
    for seed in SEEDS:
        for kind in KINDS:
            folder = source_dir(root, seed, kind)
            status_path = folder / "status.json"
            status = json.loads(status_path.read_text()).get("status", "unknown") if status_path.exists() else "queued"
            entries = curves[kind][seed]
            last = entries[-1] if entries else None
            table.append(f"<tr><td>{seed}</td><td>{kind.title()}</td><td>{status}</td>"
                         f"<td>{last['env_steps']:,}</td><td>{last['successes']}/{last['episodes']}</td></tr>"
                         if last else f"<tr><td>{seed}</td><td>{kind.title()}</td><td>{status}</td><td>—</td><td>—</td></tr>")
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='refresh' content='120'><title>Pi0 task 55 five seeds</title>"
            "<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:auto;padding:30px;color:#17313b;background:#f5f7f6}"
            "main{background:white;border-radius:14px;padding:28px}img{width:100%}table{border-collapse:collapse;width:100%}"
            "td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style></head><body><main>"
            "<h1>π₀ on LIBERO-90 task 55</h1><p>Frozen π₀ with SAC residual-noise steering. "
            "Five matched seeds compare simulator sparse reward with Robometer progress. "
            "Each evaluation uses 25 held-out initial states; dotted lines are individual seeds. "
            "The bold mean and 95% t interval use available seeds at each checkpoint, so n varies while runs finish.</p>"
            "<img src='success_rate_five_seeds.png' alt='Success rate against training transitions'>"
            "<p><a href='success_rate_five_seeds.pdf'>PDF</a> · <a href='curves.json'>Curve data</a></p>"
            "<table><thead><tr><th>Seed</th><th>Reward</th><th>Status</th><th>Latest evaluated transitions</th>"
            "<th>Simulator successes</th></tr></thead><tbody>" + "".join(table) + "</tbody></table>"
            "<p>Seed 100 completed previously. Seeds 101–104 run to 100,000 physical transitions per reward arm. "
            "The released pi0 weights remain frozen; only the SAC steering controller learns. No W&B is used.</p>"
            "</main></body></html>")
    (root / "index.html").write_text(page)


def run_one(root, gpu, kind, seed):
    folder = source_dir(root, seed, kind)
    status_path = folder / "status.json"
    process = None
    if folder.exists():
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        if state.get("status") == "complete" and (folder / "checkpoints/final.pt").exists():
            return
        job_path = folder / "job.json"
        if not job_path.exists():
            raise RuntimeError(f"Existing incomplete run has no job PID: {folder}")
        pid = json.loads(job_path.read_text())["pid"]
        if not trainer_alive(pid, folder):
            raise RuntimeError(f"Existing {kind} seed {seed} run ended incomplete; inspect {folder}")
    else:
        while True:
            try:
                assert_idle([gpu])
                break
            except RuntimeError as error:
                write_json(root / f"lane_{gpu}.json", {"status": "waiting_for_gpu", "gpu": gpu,
                                                         "kind": kind, "seed": seed,
                                                         "reason": str(error), "updated": now()})
                time.sleep(30)
        folder.mkdir(parents=True, exist_ok=False)
        base = source_dir(root, 100, kind)
        cfg = json.loads((base / "config.json").read_text())
        if cfg["seed"] != 100 or cfg["task"] != TASK or cfg["steps"] != 100000 or cfg["reward"] != kind:
            raise RuntimeError("Seed-100 source config does not match the intended experiment")
        cfg.update(seed=seed, policy_gpu=gpu, reward_gpu=gpu)
        write_json(folder / "config.json", cfg)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
               "MUJOCO_EGL_DEVICE_ID": str(gpu), "EGL_DEVICE_ID": str(gpu),
               "AUDIT_LAUNCHER_GUARDED": "1", "PYTHONPATH": source_pythonpath(),
               "WANDB_MODE": "disabled", "WANDB_DISABLED": "true",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        with (folder / "console.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.pi0_stove.rl",
                                        "--config", str(folder / "config.json"), "--out", str(folder)],
                                       cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid = process.pid
        write_json(folder / "job.json", {"pid": pid, "started": now(), "gpu": gpu})
    while trainer_alive(pid, folder):
        write_json(root / f"lane_{gpu}.json", {"status": "running", "gpu": gpu,
                                                 "kind": kind, "seed": seed, "pid": pid, "updated": now()})
        time.sleep(30)
    if process is not None and process.wait() != 0:
        raise RuntimeError(f"{kind} seed {seed} trainer exited with code {process.returncode}")
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    if state.get("status") != "complete" or state.get("env_steps") != 100000 or not (folder / "checkpoints/final.pt").exists():
        raise RuntimeError(f"{kind} seed {seed} did not finish correctly: {state}")
    write_json(root / f"lane_{gpu}.json", {"status": "job_complete", "gpu": gpu,
                                             "kind": kind, "seed": seed, "pid": pid, "updated": now()})


def lane(root, gpu):
    try:
        while True:
            try:
                reservation = reserve_gpus([gpu])
                reservation.__enter__()
                break
            except RuntimeError as error:
                write_json(root / f"lane_{gpu}.json", {"status": "waiting_for_gpu", "gpu": gpu,
                                                         "reason": str(error), "updated": now()})
                time.sleep(30)
        try:
            for kind, seed in LANES[gpu]:
                run_one(root, gpu, kind, seed)
        finally:
            reservation.__exit__(None, None, None)
        write_json(root / f"lane_{gpu}.json", {"status": "complete", "gpu": gpu, "updated": now()})
    except BaseException as error:
        write_json(root / f"lane_{gpu}.json", {"status": "failed", "gpu": gpu,
                                                 "error": repr(error), "updated": now()})
        raise


def launch(root, seed100_run):
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment: {root}")
    source = Path(seed100_run).resolve()
    for kind in KINDS:
        folder = source / f"{kind}_seed100"
        cfg = json.loads((folder / "config.json").read_text())
        status = json.loads((folder / "status.json").read_text())
        if cfg["task"] != TASK or cfg["seed"] != 100 or cfg["steps"] != 100000 or status["status"] != "complete":
            raise RuntimeError(f"Seed-100 {kind} run is not a completed matching source")
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"task": TASK, "seeds": SEEDS, "new_seeds": NEW_SEEDS,
                                       "seed100_run": str(source), "created": now(), "lanes": LANES})
    report(root)
    processes = {}
    for gpu in (0, 1):
        env = {**os.environ, "WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}
        with (root / f"lane_{gpu}.log").open("w") as log:
            processes[gpu] = subprocess.Popen(
                [sys.executable, "-u", "-m", "experiments.pi0_stove.multiseed", "lane",
                 "--root", str(root), "--gpu", str(gpu)], cwd=REPO, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(root / "orchestrator.json", {"status": "running", "pid": os.getpid(),
                                             "lane_pids": {str(g): p.pid for g, p in processes.items()},
                                             "updated": now()})
    while any(p.poll() is None for p in processes.values()):
        report(root)
        write_json(root / "orchestrator.json", {"status": "running", "pid": os.getpid(),
                                                 "lane_pids": {str(g): p.pid for g, p in processes.items()},
                                                 "lane_returncodes": {str(g): p.poll() for g, p in processes.items()},
                                                 "updated": now()})
        time.sleep(60)
    report(root)
    status = "complete" if all(p.returncode == 0 for p in processes.values()) else "failed"
    write_json(root / "orchestrator.json", {"status": status, "pid": os.getpid(),
                                             "lane_pids": {str(g): p.pid for g, p in processes.items()},
                                             "lane_returncodes": {str(g): p.returncode for g, p in processes.items()},
                                             "updated": now()})
    if status != "complete":
        raise RuntimeError("A GPU lane failed; inspect lane logs and run status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--root", required=True, type=Path)
    launch_parser.add_argument("--seed100-run", required=True, type=Path)
    lane_parser = sub.add_parser("lane")
    lane_parser.add_argument("--root", required=True, type=Path)
    lane_parser.add_argument("--gpu", required=True, type=int, choices=(0, 1))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.root.resolve(), args.seed100_run)
    elif args.command == "lane":
        lane(args.root.resolve(), args.gpu)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()

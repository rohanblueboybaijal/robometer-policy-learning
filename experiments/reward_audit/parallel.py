"""One coordinator: wait for a run, execute two GPU lanes, and update the HTML report."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO, load_suite, materialize, write_json
from .html_report import generate_html
from .launch import preflight, reserve_gpus, start_run, stop_run


def make_plan(suite, seeds):
    base = [materialize(suite, run) for run in suite["runs"]]
    if len(base) != 2 or sorted(c["reward"] for c in base) != ["robometer", "sparse"]:
        raise ValueError("Suite must contain one sparse arm and one Robometer arm")
    if any(c["audit_sparse_rewards"] for c in base):
        raise ValueError("Sparse arm must not load a reward model")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return [dict(cfg, seed=seed, name=f"{cfg['name']}_seed{seed}") for seed in seeds for cfg in base]


def prerequisite_ready(folder):
    if folder is None:
        return True
    path = Path(folder) / "status.json"
    status = json.loads(path.read_text()).get("status") if path.exists() else "pending"
    if status == "failed":
        raise RuntimeError(f"Prerequisite failed: {folder}; inspect its console.log")
    return status == "complete"


def execute(plan, out, oracle_gpu, model_gpu, after=None):
    """At most one run per GPU; each lane advances independently through seeds."""
    lanes = {kind: [cfg for cfg in plan if cfg["reward"] == kind] for kind in ("sparse", "robometer")}
    running = {}
    previous = out / "experiment_status.json"
    records = json.loads(previous.read_text()).get("runs", {}) if previous.exists() else {}
    records.update({cfg["name"]: {"seed": cfg["seed"], "reward": cfg["reward"], "status": "pending"} for cfg in plan})
    failed = {r["reward"] for r in records.values() if r["status"] == "failed"}
    indices = [oracle_gpu if cfg["reward"] == "sparse" else model_gpu for cfg in plan]

    def state(status, **extra):
        write_json(
            out / "experiment_status.json",
            {
                "status": status,
                "pid": os.getpid(),
                "updated": datetime.now(timezone.utc).isoformat(),
                "after_run": str(after) if after else None,
                "runs": records,
                **extra,
            },
        )

    def refresh():
        try:
            generate_html(out)
        except (ValueError, OSError, KeyError) as error:
            print(f"Report refresh deferred: {error}", flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    refresh()
    try:
        with ExitStack() as stack:
            while True:
                if not prerequisite_ready(after):
                    state("waiting_for_prerequisite")
                else:
                    try:
                        stack.enter_context(reserve_gpus(indices))
                        break
                    except RuntimeError as error:
                        state("waiting_for_gpus", reason=str(error))
                time.sleep(30)
            # This checks dependencies/storage without importing ML modules.
            if set(indices) == {model_gpu}:
                checks = preflight(plan, sys.executable, model_gpu, model_gpu, out, True)
            else:
                checks = preflight(plan, sys.executable, oracle_gpu, model_gpu, out)
            write_json(out / "preflight.json", checks)
            if not checks["ready"]:
                raise RuntimeError(f"Preflight failed: {checks['errors']}")
            last_report = -float("inf")
            while any(lanes.values()) or running:
                for kind in lanes:
                    if kind in running:
                        process, cfg = running[kind]
                        if process.poll() is None:
                            continue
                        del running[kind]
                        ok = process.returncode == 0 and prerequisite_ready(out / cfg["name"])
                        records[cfg["name"]]["status"] = "complete" if ok else "failed"
                        if not ok:
                            failed.add(kind)
                    if lanes[kind]:
                        cfg = lanes[kind].pop(0)
                        gpu = oracle_gpu if kind == "sparse" else model_gpu
                        process = start_run(cfg, out / cfg["name"], gpu, gpu, sys.executable)
                        running[kind] = (process, cfg)
                        records[cfg["name"]].update(status="running", pid=process.pid, gpu=gpu)
                state("running")
                if time.monotonic() - last_report >= 300:
                    refresh()
                    last_report = time.monotonic()
                if any(lanes.values()) or running:
                    time.sleep(30)
            state("failed" if failed else "complete")
            refresh()
    except KeyboardInterrupt:
        state("cancelled")
        raise
    except Exception as error:
        state("failed", error=str(error))
        raise
    finally:
        for process, _ in running.values():
            stop_run(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--after-run", type=Path)
    parser.add_argument("--include-existing", type=Path, help="Completed original pair to include in statistics")
    parser.add_argument("--oracle-gpu", type=int, default=1)
    parser.add_argument("--model-gpu", type=int, default=0)
    args = parser.parse_args()
    if args.oracle_gpu == args.model_gpu:
        parser.error("Assign different GPUs to the two lanes")
    suite = load_suite(args.suite.resolve())
    plan = make_plan(suite, args.seeds)
    out = args.out.resolve()
    after = args.after_run.resolve() if args.after_run else None
    existing = args.include_existing.resolve() if args.include_existing else None
    os.chdir(REPO)
    out.mkdir(parents=True, exist_ok=False)
    if existing:
        for path in sorted(existing.glob("*/config.json")):
            cfg = json.loads(path.read_text())
            if cfg["seed"] in args.seeds:
                raise ValueError("Additional seeds overlap the original seed")
            (out / f"{cfg['name']}_seed{cfg['seed']}").symlink_to(path.parent, target_is_directory=True)
    write_json(out / "planned_suite.json", {"description": suite["description"], "runs": plan})
    execute(plan, out, args.oracle_gpu, args.model_gpu, after)


if __name__ == "__main__":
    main()

"""Wait for prerequisite runs and genuinely idle GPUs, then use the guarded launcher."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from .config import HERE, REPO, load_suite, materialize, write_json
from .gpu_guard import assert_idle


def prerequisites(out, names):
    pending = []
    for name in names:
        path = Path(out) / name / "status.json"
        status = json.loads(path.read_text()).get("status") if path.exists() else "pending"
        if status == "failed":
            raise RuntimeError(f"Prerequisite {name} failed; inspect its console.log")
        if status != "complete":
            pending.append(name)
    return pending


def required_gpus(suite_path, names, policy_gpu, reward_gpu):
    suite = load_suite(suite_path)
    available = {run["name"] for run in suite["runs"]}
    if set(names) - available:
        raise ValueError(f"Unknown runs: {set(names) - available}")
    configs = [materialize(suite, run) for run in suite["runs"] if run["name"] in names]
    needs_reward = any(cfg["reward"] == "robometer" or cfg["audit_sparse_rewards"] for cfg in configs)
    return [policy_gpu, reward_gpu] if needs_reward else [policy_gpu]


def assert_unreserved(indices):
    # A launcher reserves its GPU while a worker is loading weights on CPU,
    # and between its sequential arms. Allocation can briefly be zero then.
    with ExitStack() as stack:
        for index in sorted(set(indices)):
            lock = stack.enter_context(open(Path(tempfile.gettempdir()) / f"reward-audit-gpu-{index}.lock", "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another reward-audit launcher has reserved GPU {index}") from None


def refresh_curves(out):
    from .report import generate_report

    try:
        generate_report(out)
    except (ValueError, KeyError, OSError, TypeError) as error:
        # A writer may be halfway through flushing the newest CSV/JSONL row.
        print(f"Curve refresh deferred: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=HERE / "configs/libero.json")
    parser.add_argument("--after", nargs="+", default=[], help="Optional prerequisite runs in --out")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--policy-gpu", type=int, required=True)
    parser.add_argument("--reward-gpu", type=int, required=True)
    parser.add_argument("--allow-shared-reward-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--idle-polls", type=int, default=3)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.poll_seconds > 60 or args.idle_polls < 1:
        parser.error("Use 1-60 second polling and at least one idle observation")
    os.chdir(REPO)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    indices = required_gpus(args.suite, args.runs, args.policy_gpu, args.reward_gpu)
    state_path = args.out / "reward_queue.json"

    def state(status, **extra):
        write_json(
            state_path,
            dict(
                status=status,
                pid=os.getpid(),
                updated=datetime.now(timezone.utc).isoformat(),
                runs=args.runs,
                required_gpus=indices,
                **extra,
            ),
        )

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    child = None
    last_plot = -float("inf")
    idle = 0
    try:
        while idle < args.idle_polls:
            pending = prerequisites(args.out, args.after)
            reason = f"Waiting for {pending}" if pending else None
            if not pending:
                try:
                    assert_idle(indices)
                    assert_unreserved(indices)
                except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
                    reason = str(error)
            idle = 0 if reason else idle + 1
            state("waiting", reason=reason, idle_observations=idle)
            print(reason or f"Required GPUs {indices} idle: observation {idle}/{args.idle_polls}", flush=True)
            if time.monotonic() - last_plot > 300:
                refresh_curves(args.out)
                last_plot = time.monotonic()
            if idle < args.idle_polls:
                time.sleep(args.poll_seconds)
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.reward_audit.launch",
            "--suite",
            str(args.suite),
            "--runs",
            *args.runs,
            "--out",
            str(args.out),
            "--policy-gpu",
            str(args.policy_gpu),
            "--reward-gpu",
            str(args.reward_gpu),
            "--policy-python",
            sys.executable,
        ]
        if args.allow_shared_reward_gpu:
            command.append("--allow-shared-reward-gpu")
        # The launcher independently rechecks devices and acquires its GPU locks.
        with (args.out / "reward_driver.log").open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )
            state("running", launcher_pid=child.pid, command=command)
            while child.poll() is None:
                if time.monotonic() - last_plot > 300:
                    refresh_curves(args.out)
                    last_plot = time.monotonic()
                time.sleep(args.poll_seconds)
        if child.returncode:
            raise RuntimeError(f"Reward launcher exited {child.returncode}; inspect reward_driver.log")
        refresh_curves(args.out)
        state("complete")
    except KeyboardInterrupt:
        state("cancelled")
        raise
    except Exception as error:
        state("failed", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            # This is only our own launcher; it handles SIGTERM and cleans up
            # its own newly created experiment process group.
            child.terminate()
            child.wait(timeout=60)


if __name__ == "__main__":
    main()

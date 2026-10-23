"""Only this entrypoint launches real experiments. Checks happen before ML imports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .config import HERE, REPO, load_suite, materialize, write_json
from .gpu_guard import assert_idle, snapshot


def source_pythonpath():
    # LIBERO's outer package is a namespace, which its legacy find_packages()
    # setup does not include in an editable installation.
    return os.pathsep.join([str(REPO), str(REPO / "LIBERO"), os.environ.get("PYTHONPATH", "")])


def python_check(interpreter, modules):
    interpreter = str(Path(interpreter).absolute())
    if not Path(interpreter).exists():
        return {"python": interpreter, "error": "Interpreter does not exist"}
    script = (
        "import sys,importlib.util,json; "
        "print(json.dumps({'version':sys.version,"
        "'missing':[m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]}))"
    )
    # find_spec doesn't import these top-level modules or initialize GPUs.
    result = subprocess.run(
        [interpreter, "-c", script, *modules],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": source_pythonpath()},
    )
    if result.returncode:
        return {"python": interpreter, "error": result.stderr}
    return {"python": interpreter, **json.loads(result.stdout)}


def provenance():
    result = {}
    for name in (".", "LIBERO", "dsrl_openpi", "robometer"):
        directory = REPO / name
        sha = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
        result[name] = sha.stdout.strip() if sha.returncode == 0 else None
    result["policy_diff"] = subprocess.check_output(["git", "-C", str(REPO), "diff", "--stat"], text=True)
    result["robometer_diff"] = subprocess.check_output(["git", "-C", str(REPO / "robometer"), "diff"], text=True)
    result["audit_source_sha256"] = {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(HERE.rglob("*.py"))
    }
    return result


@contextmanager
def reserve_gpus(indices):
    """Reserve our devices once; never alter another user's processes."""
    with ExitStack() as stack:
        for index in sorted(set(indices)):
            lock = stack.enter_context(open(Path(tempfile.gettempdir()) / f"reward-audit-gpu-{index}.lock", "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another reward-audit launcher has reserved GPU {index}") from None
        assert_idle(indices)
        yield


def start_run(cfg, out, policy_gpu, reward_gpu, policy_python):
    """Start one training process; caller holds the GPU reservation."""
    needs_reward = cfg["reward"] == "robometer" or cfg["audit_sparse_rewards"]
    indices = [policy_gpu, reward_gpu] if needs_reward else [policy_gpu]
    assert_idle(indices)
    out = Path(out).resolve()
    out.mkdir()  # Refuse to overwrite a previous run.
    cfg = {**cfg, "reward_python": str(Path(cfg["reward_python"]).absolute())}
    write_json(out / "config.json", cfg)
    write_json(out / "provenance.json", provenance())
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(policy_gpu),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(policy_gpu),
        "EGL_DEVICE_ID": str(policy_gpu),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false",
        "AUDIT_LAUNCHER_GUARDED": "1",
        "PYTHONPATH": source_pythonpath(),
    }
    command = [
        str(Path(policy_python).absolute()),
        "-u",
        "-m",
        "experiments.reward_audit.train",
        "--config",
        str(out / "config.json"),
        "--out",
        str(out),
        "--reward-gpu",
        str(reward_gpu),
    ]
    with (out / "console.log").open("w") as log:
        return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_run(process):
    """Stop only the newly created training process group and its children."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def preflight(runs, policy_python, policy_gpu, reward_gpu, out, allow_shared_reward_gpu=False):
    errors = []
    needs_reward = any(c["reward"] == "robometer" or c["audit_sparse_rewards"] for c in runs)
    if needs_reward and policy_gpu == reward_gpu and not allow_shared_reward_gpu:
        errors.append("Assign separate policy and reward GPUs")
    try:
        devices = snapshot()
        assert_idle([policy_gpu, reward_gpu] if needs_reward else [policy_gpu], devices)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
        errors.append(str(e))
        devices = locals().get("devices", [])
    checks = [
        python_check(
            policy_python,
            ["torch", "numpy", "gymnasium", "transformers", "libero", "robosuite", "yaml", "imageio", "loguru"],
        )
    ]
    if needs_reward:
        for interpreter in sorted({c["reward_python"] for c in runs}):
            checks.append(
                python_check(
                    interpreter, ["torch", "robometer", "unsloth", "transformers", "peft", "accelerate", "fastapi"]
                )
            )
    for check in checks:
        if check.get("error") or check.get("missing"):
            errors.append(f"Environment incomplete: {check}")
    parent = Path(out).resolve()
    while not parent.exists():
        parent = parent.parent
    free_gb = shutil.disk_usage(parent).free / 2**30
    # Conservative estimate for fresh environments + weights + local run files.
    if free_gb < (30 if needs_reward else 5):
        errors.append(f"Only {free_gb:.1f} GiB free; need {'30' if needs_reward else '5'} GiB")
    return {
        "ready": not errors,
        "errors": errors,
        "gpus": devices,
        "disk_free_gib": free_gb,
        "environments": checks,
        "provenance": provenance(),
        "runs": [
            {
                "name": c["name"],
                "task_id": c["task_id"],
                "task": c["task"],
                "reward": c["reward"],
                "steps": c["steps"],
                "num_envs": c["num_envs"],
            }
            for c in runs
        ],
    }


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=HERE / "configs/libero.json")
    parser.add_argument("--runs", nargs="+", help="Explicit run names; default is all configured arms")
    parser.add_argument("--out", type=Path, default=REPO / "outputs/reward_audit")
    parser.add_argument("--policy-python", default=".venv-policy/bin/python")
    parser.add_argument("--reward-python", help="Override reward interpreter from suite")
    parser.add_argument("--reward-checkpoint", help="HF repo@commit or local Robometer checkpoint")
    parser.add_argument("--initial-actor", help="Same native SAC actor checkpoint for all selected arms")
    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--reward-gpu", type=int, default=1)
    parser.add_argument(
        "--allow-shared-reward-gpu",
        action="store_true",
        help="Allow policy/rendering and reward inference to share one initially idle GPU",
    )
    parser.add_argument("--check", action="store_true", help="Save read-only preflight; never initialize CUDA")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configs only; no ML imports")
    args = parser.parse_args()
    os.chdir(REPO)
    suite = load_suite(args.suite)
    available = {r["name"] for r in suite["runs"]}
    if args.runs and set(args.runs) - available:
        parser.error(f"Unknown runs: {set(args.runs) - available}")
    runs = [materialize(suite, r) for r in suite["runs"] if not args.runs or r["name"] in args.runs]
    for cfg in runs:
        if args.reward_checkpoint:
            cfg["reward_checkpoint"] = args.reward_checkpoint
        if args.reward_python:
            cfg["reward_python"] = args.reward_python
        cfg["reward_python"] = str(Path(cfg["reward_python"]).absolute())
        if args.initial_actor:
            cfg["initial_actor"] = str(Path(args.initial_actor).resolve())
        # Validate overrides as well, including accidental LIBERO-90 leakage.
        materialize({"defaults": {}}, cfg)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    report = preflight(
        runs,
        args.policy_python,
        args.policy_gpu,
        args.reward_gpu,
        args.out,
        allow_shared_reward_gpu=args.allow_shared_reward_gpu,
    )
    report["allow_shared_reward_gpu"] = args.allow_shared_reward_gpu
    write_json(args.out / "preflight.json", report)
    print(json.dumps(report, indent=2))
    if args.check:
        raise SystemExit(0 if report["ready"] else 2)
    if not report["ready"]:
        raise SystemExit("Preflight failed. No experiment started.")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "suite.json", {"description": suite["description"], "runs": runs})
    needs_reward = any(c["reward"] == "robometer" or c["audit_sparse_rewards"] for c in runs)
    indices = [args.policy_gpu, args.reward_gpu] if needs_reward else [args.policy_gpu]
    # Coordinate our own launchers. This cannot reserve GPUs against unrelated jobs;
    # recheck immediately before each run and never stop anyone else's process.
    with reserve_gpus(indices):
        for cfg in runs:
            out = args.out / cfg["name"]
            process = start_run(cfg, out, args.policy_gpu, args.reward_gpu, args.policy_python)
            try:
                result = process.wait()
            except KeyboardInterrupt:
                stop_run(process)
                raise
            if result:
                raise SystemExit(f"{cfg['name']} failed; see {out}/console.log")
            print(f"Completed {cfg['name']}. Curves: {out}/evaluation.csv", flush=True)
    from .report import generate_report

    generate_report(args.out)


if __name__ == "__main__":
    main()

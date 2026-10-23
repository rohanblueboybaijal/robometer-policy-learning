from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def task_map():
    # Read metadata without importing torch, LIBERO or creating a renderer.
    path = REPO / "LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text())
    return ast.literal_eval(tree.body[0].value)


def resolve_task(suite, name):
    names = task_map()[suite]
    if name not in names:
        raise ValueError(f"Unknown {suite} task: {name}")
    return names.index(name)


def load_suite(path):
    cfg = json.loads(Path(path).read_text())
    names = set()
    for run in cfg["runs"]:
        if run["name"] in names:
            raise ValueError("Run names must be unique")
        names.add(run["name"])
        materialize(cfg, run)
    return cfg


def materialize(suite, run):
    cfg = copy.deepcopy(suite["defaults"])
    for key, value in run.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    cfg["task_id"] = resolve_task(cfg["suite"], cfg["task"])
    if cfg["reward"] not in {"sparse", "robometer"}:
        raise ValueError("Reward must be sparse or robometer")
    if cfg["reward_camera"] not in {"agentview_image", "robot0_eye_in_hand_image"}:
        raise ValueError("Unknown reward camera")
    if cfg["reward_formula"] not in {"progress", "progress_minus_one", "sparse_plus_progress", "delta"}:
        raise ValueError("Unknown reward formula")
    for key in ("num_envs", "steps", "eval_every", "eval_episodes", "horizon", "max_frames", "updates_per_transition"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["steps"] % cfg["num_envs"]:
        raise ValueError("steps must be divisible by num_envs (aggregate physical steps)")
    if cfg["train_init_states"] < 1:
        raise ValueError("Need at least one training reset state")
    if "fft-libero" in cfg["reward_checkpoint"] and cfg.get("ood_claim") == "held_out_task":
        raise ValueError("Robometer-LIBERO includes LIBERO-90: cannot claim held-out tasks")
    return cfg


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)

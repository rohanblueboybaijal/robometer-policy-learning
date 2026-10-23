"""Evaluate pretrained pi0 on 25 held-out LIBERO-Goal initial states."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from experiments.reward_audit.config import HERE, REPO, load_suite, materialize, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.gpu_guard import assert_idle
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.metrics import wilson


class Pi0Client:
    def __init__(self, interpreter, checkpoint, seed, gpu, directory, config_name="pi0_libero"):
        self.process = None
        self.connection = None
        self.temp = tempfile.TemporaryDirectory(prefix="pi0-audit-")
        self.socket = str(Path(self.temp.name) / "policy.sock")
        if len(self.socket.encode()) >= 107:
            raise ValueError("Pi0 RPC socket path too long; choose a shorter TMPDIR")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "AUDIT_PI0_RPC_KEY": os.urandom(32).hex(),
            "PYTHONPATH": source_pythonpath(),
        }
        self.log = (directory / "policy_worker.log").open("w")
        try:
            self.process = subprocess.Popen(
                [str(interpreter), "-u", "-m", "experiments.pi0_stove.policy_worker", "--socket", self.socket,
                 "--checkpoint", str(checkpoint), "--seed", str(seed), "--config", config_name],
                env=env, cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 1800
            while not Path(self.socket).exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Pi0 worker failed; see {directory / 'policy_worker.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Pi0 worker did not initialize")
                time.sleep(.2)
            self.connection = Client(self.socket, family="AF_UNIX", authkey=bytes.fromhex(env["AUDIT_PI0_RPC_KEY"]))
            self.metadata = self.connection.recv()
        except BaseException:
            self.close()
            raise

    def infer(self, observation, prompt, noise=None, noise_mode="additive"):
        if not self.connection:
            raise RuntimeError("Pi0 worker is closed")
        images = {k: np.ascontiguousarray(observation[k], dtype=np.uint8).tobytes()
                  for k in ("agentview_image", "robot0_eye_in_hand_image")}
        self.connection.send({"observation": {"state": observation["state"].tolist()},
                              "images": images, "prompt": prompt,
                              "noise": None if noise is None else np.asarray(noise, dtype=np.float32).tolist(),
                              "noise_mode": noise_mode})
        if not self.connection.poll(600):
            raise TimeoutError("Pi0 inference timed out")
        actions = np.asarray(self.connection.recv(), dtype=np.float32)
        expected = (self.metadata["action_horizon"], 7)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Invalid Pi0 response: {actions.shape}")
        return actions

    def reset_seed(self, seed):
        self.connection.send({"seed": int(seed)})
        if not self.connection.poll(30) or self.connection.recv() is not True:
            raise RuntimeError("Pi0 worker did not reset its sample RNG")

    def close(self):
        if self.connection:
            try:
                self.connection.send(None)
            except (EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=30)
            self.process = None
        self.log.close()
        self.temp.cleanup()


def evaluate(policy, env, cfg, directory, action_exec_len):
    language = env.metadata[0]["language"]
    records = []
    with (directory / "episodes.jsonl").open("w", buffering=1) as stream:
        for index in range(cfg["train_init_states"], cfg["train_init_states"] + cfg["eval_episodes"]):
            observation = env.reset({0: index})[0]["obs"]
            actions_generated = 0
            while True:
                actions = policy.infer(observation, language)
                actions_generated += 1
                finished = False
                for action in actions[:action_exec_len]:
                    result = env.step([action])[0]
                    observation = result["obs"]
                    if result["terminated"] or result["truncated"]:
                        finished = True
                        break
                if finished:
                    record = {"init_state_id": index, "success": bool(result["terminated"]),
                              "length": result["elapsed"], "chunks": actions_generated}
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    print(f"State {index}: {'success' if record['success'] else 'failure'} "
                          f"after {record['length']} steps", flush=True)
                    break
    successes = sum(record["success"] for record in records)
    return {"successes": successes, "episodes": len(records), "success_rate": successes / len(records),
            "wilson_ci95": wilson(successes, len(records)), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi0-python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--action-exec-len", type=int, default=20)
    parser.add_argument("--suite-config", type=Path, default=HERE / "configs/stove.json")
    args = parser.parse_args()
    suite = load_suite(args.suite_config)
    cfg = materialize(suite, suite["runs"][0])
    os.environ["PYTHONPATH"] = source_pythonpath()
    if not 1 <= args.action_exec_len <= 50:
        parser.error("action-exec-len must be between 1 and 50")
    if not (args.checkpoint / "params").exists() or not (args.checkpoint / "assets").exists():
        parser.error("Pi0 checkpoint needs params and assets")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"task": cfg["task"], "task_id": cfg["task_id"],
                                      "checkpoint": str(args.checkpoint.resolve()), "seed": cfg["seed"],
                                      "action_exec_len": args.action_exec_len,
                                      "eval_state_ids": list(range(cfg["train_init_states"],
                                                                   cfg["train_init_states"] + cfg["eval_episodes"])),
                                      "horizon": cfg["horizon"], "gpu": args.gpu})
    prepare_libero_paths(output / "libero_paths", REPO)
    policy = env = None
    try:
        with reserve_gpus([args.gpu]):
            assert_idle([args.gpu])
            # Keep the venv symlink: resolving it selects the bare interpreter without site-packages.
            policy = Pi0Client(args.pi0_python.absolute(), args.checkpoint.resolve(), cfg["seed"], args.gpu, output)
            env = EnvPool(cfg, 1, rank_offset=100000)
            result = evaluate(policy, env, cfg, output, args.action_exec_len)
            write_json(output / "summary.json", result)
            write_json(output / "status.json", {"status": "complete", "success_rate": result["success_rate"]})
            print(f"Pi0 baseline: {result['successes']}/{result['episodes']} = {result['success_rate']:.1%}", flush=True)
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        if env:
            env.close()
        if policy:
            policy.close()


if __name__ == "__main__":
    main()

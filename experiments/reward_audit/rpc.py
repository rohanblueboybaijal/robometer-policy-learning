"""Private Unix-socket RPC for a reward process in its own uv environment/GPU."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path


class RewardClient:
    def __init__(self, cfg, out, gpu):
        self.connection = None
        self.process = None
        self.temp = tempfile.TemporaryDirectory(prefix="rbm-audit-")
        self.socket = str(Path(self.temp.name) / "reward.sock")
        self.timeout = cfg["worker_timeout"]
        self.log = (Path(out) / "reward_worker.log").open("w")
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            WANDB_MODE="disabled",
            WANDB_DISABLED="true",
            TOKENIZERS_PARALLELISM="false",
            AUDIT_RPC_KEY=os.urandom(32).hex(),
        )
        try:
            self.process = subprocess.Popen(
                [
                    cfg["reward_python"],
                    "-m",
                    "experiments.reward_audit.reward_worker",
                    "--socket",
                    self.socket,
                    "--config",
                    str(Path(out) / "config.json"),
                    "--out",
                    str(out),
                ],
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + self.timeout
            while not Path(self.socket).exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Reward worker failed; see {out}/reward_worker.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Reward worker failed to start")
                time.sleep(0.1)
            self.connection = Client(self.socket, family="AF_UNIX", authkey=bytes.fromhex(env["AUDIT_RPC_KEY"]))
            self.metadata = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError("Reward inference timed out; no placeholder rewards will be used")
        response = self.connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def predict(self, frames, task):
        self.connection.send({"frames": frames, "task": task})
        return self._receive()

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(None)
            except (EOFError, OSError):
                pass
            self.connection.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()
        self.temp.cleanup()

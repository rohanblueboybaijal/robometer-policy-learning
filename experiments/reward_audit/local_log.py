from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class LocalLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics = (self.directory / "metrics.jsonl").open("a", buffering=1)
        self.episodes = (self.directory / "episodes.jsonl").open("a", buffering=1)
        self.curves = (self.directory / "evaluation.csv").open("a", newline="", buffering=1)
        self.writer = csv.DictWriter(
            self.curves,
            fieldnames=["env_steps", "successes", "episodes", "success_rate", "ci_low", "ci_high", "elapsed_seconds"],
        )
        if self.curves.tell() == 0:
            self.writer.writeheader()
        self.start = time.monotonic()

    def log(self, values, step=None, prefix=None, **kwargs):
        self.metrics.write(
            json.dumps(
                {
                    "env_steps": step,
                    "prefix": prefix,
                    "elapsed_seconds": time.monotonic() - self.start,
                    "metrics": values,
                },
                allow_nan=False,
            )
            + "\n"
        )

    def episode(self, data):
        self.episodes.write(json.dumps(data, allow_nan=False) + "\n")

    def evaluation(self, step, successes, episodes, ci):
        self.writer.writerow(
            dict(
                env_steps=step,
                successes=successes,
                episodes=episodes,
                success_rate=successes / episodes,
                ci_low=ci[0],
                ci_high=ci[1],
                elapsed_seconds=time.monotonic() - self.start,
            )
        )

    def close(self):
        for f in (self.metrics, self.episodes, self.curves):
            f.close()

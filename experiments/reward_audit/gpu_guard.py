"""Read-only GPU checks. No override for an occupied device."""

from __future__ import annotations

import csv
import io
import subprocess


def snapshot():
    def query(args):
        return list(
            csv.reader(
                io.StringIO(subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True))
            )
        )

    devices = query(["--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu"])
    processes = query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"])
    return [
        dict(
            index=int(r[0]),
            uuid=r[1].strip(),
            name=r[2].strip(),
            used_mb=int(r[3]),
            free_mb=int(r[4]),
            utilization=int(r[5]),
            processes=[
                {"pid": p[1].strip(), "name": p[2].strip(), "mb": p[3].strip()}
                for p in processes
                if p and p[0].strip() == r[1].strip()
            ],
        )
        for r in devices
        if r
    ]


def assert_idle(indices, devices=None):
    devices = snapshot() if devices is None else devices
    by_id = {d["index"]: d for d in devices}
    for index in set(indices):
        if index not in by_id:
            raise RuntimeError(f"GPU {index} does not exist")
        d = by_id[index]
        # memory check also detects graphics/rendering jobs not in compute-apps.
        if d["processes"] or d["used_mb"] > 512 or d["utilization"] > 5:
            raise RuntimeError(
                f"GPU {index} is occupied ({d['used_mb']} MiB, {d['utilization']}% utilization). No experiment started."
            )
    return devices

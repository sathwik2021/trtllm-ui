from __future__ import annotations

import csv
import io
import subprocess
import time


def poll() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        if p.returncode != 0:
            return {"error": (p.stderr or p.stdout or "nvidia-smi failed").strip()}
        gpus = []
        for idx, row in enumerate(csv.reader(io.StringIO(p.stdout), skipinitialspace=True)):
            if len(row) < 5:
                continue
            gpus.append({
                "index": idx,
                "utilization_gpu": row[0].strip(),
                "memory_used": row[1].strip(),
                "memory_total": row[2].strip(),
                "temperature_gpu": row[3].strip(),
                "power_draw": row[4].strip(),
            })
        return {"gpus": gpus}
    except Exception as exc:
        return {"error": str(exc)}


def poll_with_retry(attempts: int = 3, delay: float = 3.0) -> dict:
    """Like poll(), but retries a few times before giving up.

    Intended for use at app startup only (see main.py's startup event),
    not for the live /api/gpu polling loop -- WSL2's GPU passthrough can
    have a brief delay right after Docker Desktop starts on a cold boot,
    and a single nvidia-smi failure at t=0 shouldn't be reported as
    "GPU permanently unavailable".
    """
    result = {"error": "not polled"}
    for attempt in range(attempts):
        result = poll()
        if "error" not in result:
            return result
        if attempt < attempts - 1:
            time.sleep(delay)
    return result

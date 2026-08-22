from __future__ import annotations

"""
Benchmark Manager (Phase 3).

Methodology deliberately mirrors this project's existing Ollama-comparison
approach so results are comparable across backends:
- fixed request count
- fixed target completion tokens (max_tokens in the request itself, not a
  token-counting heuristic after the fact)
- wall-clock timer around the FULL batch (not per-request summed)
- aggregate throughput = total completion tokens / wall time
- concurrency N fired against /v1/chat/completions, same shape as the
  Ollama parallel test
- GPU utilization/VRAM sampled on a background poll loop for the run's
  duration; min/max/avg reported
- p50/p95 per-request latency computed from a stored raw array, not
  deferred -- this is a real computed output every run, not a nice-to-have

Uses a thread pool + blocking urllib requests rather than asyncio/httpx to
avoid adding a new dependency -- this project's own health-check code
(deployment_manager._watch) already uses blocking urllib the same way, and
plain OS threads give the same real concurrent load on the server as
asyncio.gather would for this purpose (each thread blocks on I/O, not CPU).
"""

import concurrent.futures
import json
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from .settings import AppSettings, to_host_path
from . import gpu_monitor

DEFAULT_PROMPT = (
    "Write a short paragraph about the history of coffee, covering its "
    "origins and how it spread around the world."
)


class BenchmarkRequest(BaseModel):
    # Either point at a running deployment by id (host/port/served_model_name
    # get resolved from it), or supply host/port/served_model_name directly
    # for a deployment not tracked by this app's in-memory state.
    deployment_id: str | None = None
    host: str | None = None
    port: int | None = None
    served_model_name: str | None = None

    request_count: int = 20
    concurrency: int = 1
    max_tokens: int = 256
    prompt: str | None = None
    request_timeout: float = 120.0

_job_lock = threading.Lock()
_current_job: dict[str, Any] | None = None


def _benchmarks_dir(settings: AppSettings) -> Path:
    d = to_host_path(settings.data_dir) / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _one_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    """Fire one /v1/chat/completions request, block until complete.

    Never raises -- a failed request comes back as a dict with "error"
    set, so one bad request doesn't abort the whole concurrent batch or
    skew the wall-clock timer for the others still in flight.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.monotonic()
    try:
        with urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        latency = time.monotonic() - start
        usage = body.get("usage", {}) or {}
        return {
            "latency_s": latency,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "error": None,
        }
    except Exception as exc:
        return {
            "latency_s": time.monotonic() - start,
            "completion_tokens": None, "prompt_tokens": None,
            "error": str(exc),
        }


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile. Returns None on an empty list --
    callers must not report a fabricated 0 for "no successful requests".
    """
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_benchmark(
    *,
    host: str,
    port: int,
    served_model_name: str,
    request_count: int = 20,
    concurrency: int = 1,
    max_tokens: int = 256,
    prompt: str | None = None,
    request_timeout: float = 120.0,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    """Blocking. Run request_count requests, `concurrency` at a time,
    against a live deployment's /v1/chat/completions. Intended to be
    called from a background thread (see start_benchmark_job), not
    directly inside an API request handler -- a real run can take
    anywhere from seconds to several minutes.
    """
    base_url = f"http://{host}:{port}"
    prompt = prompt or DEFAULT_PROMPT
    request_count = max(1, request_count)
    concurrency = max(1, concurrency)

    gpu_samples: list[dict[str, Any]] = []
    stop_polling = threading.Event()

    def _poll_gpu() -> None:
        while not stop_polling.is_set():
            sample = gpu_monitor.poll()
            if sample.get("gpus"):
                gpu_samples.append(sample["gpus"][0])
            stop_polling.wait(1.0)

    poll_thread = threading.Thread(target=_poll_gpu, daemon=True)
    poll_thread.start()

    results: list[dict[str, Any]] = []
    wall_start = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_one_request, base_url, served_model_name, prompt, max_tokens, request_timeout)
                for _ in range(request_count)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    finally:
        wall_time_s = time.monotonic() - wall_start
        stop_polling.set()
        poll_thread.join(timeout=2.0)

    ok = [r for r in results if not r["error"]]
    errors = [r["error"] for r in results if r["error"]]
    latencies = [r["latency_s"] for r in ok]
    total_completion_tokens = sum(r["completion_tokens"] for r in ok if r["completion_tokens"])

    def _numeric(key: str) -> list[float]:
        vals = []
        for s in gpu_samples:
            v = s.get(key)
            if v in (None, ""):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        return vals

    util_vals = _numeric("utilization_gpu")
    mem_vals = _numeric("memory_used")

    return {
        "id": benchmark_id or str(uuid.uuid4()),
        "created_at": time.time(),
        "config": {
            "host": host, "port": port, "served_model_name": served_model_name,
            "request_count": request_count, "concurrency": concurrency,
            "max_tokens": max_tokens, "prompt": prompt,
        },
        "wall_time_s": wall_time_s,
        "requests_ok": len(ok),
        "requests_failed": len(errors),
        # Capped: this is a diagnostic sample, not a full error log --
        # storing unbounded error text for a large failed batch would
        # bloat the saved result file for no added diagnostic value.
        "errors": errors[:10],
        "total_completion_tokens": total_completion_tokens,
        "throughput_tokens_per_s": (total_completion_tokens / wall_time_s) if wall_time_s > 0 else None,
        "latency_s": {
            "raw": latencies,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "avg": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "gpu": {
            "samples_captured": len(gpu_samples),
            "utilization_gpu_pct": {
                "min": min(util_vals) if util_vals else None,
                "max": max(util_vals) if util_vals else None,
                "avg": statistics.mean(util_vals) if util_vals else None,
            },
            "memory_used_mb": {
                "min": min(mem_vals) if mem_vals else None,
                "max": max(mem_vals) if mem_vals else None,
                "avg": statistics.mean(mem_vals) if mem_vals else None,
            },
        },
    }


def save_result(settings: AppSettings, report: dict[str, Any]) -> Path:
    path = _benchmarks_dir(settings) / f"{report['id']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def list_results(settings: AppSettings) -> list[dict[str, Any]]:
    d = _benchmarks_dir(settings)
    out = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def get_result(settings: AppSettings, benchmark_id: str) -> dict[str, Any]:
    path = _benchmarks_dir(settings) / f"{benchmark_id}.json"
    if not path.exists():
        raise KeyError(f"Benchmark not found: {benchmark_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_result(settings: AppSettings, benchmark_id: str) -> None:
    path = _benchmarks_dir(settings) / f"{benchmark_id}.json"
    if path.exists():
        path.unlink()


def current_job() -> dict[str, Any] | None:
    with _job_lock:
        return dict(_current_job) if _current_job else None


def start_benchmark_job(settings: AppSettings, **kwargs: Any) -> dict[str, Any]:
    """Non-blocking. Starts the benchmark on a background thread and
    returns immediately with a job id -- same async-job shape as
    capability_probe's run_probe/cached_manifest pair. Only one benchmark
    runs at a time (guarded by _job_lock): concurrent runs would contend
    for the same GPU and skew both runs' numbers.
    """
    global _current_job
    with _job_lock:
        if _current_job is not None and _current_job.get("status") == "running":
            raise RuntimeError("A benchmark is already running -- wait for it to finish first")
        benchmark_id = str(uuid.uuid4())
        _current_job = {"id": benchmark_id, "status": "running", "started_at": time.time()}

    def _run() -> None:
        global _current_job
        try:
            report = run_benchmark(benchmark_id=benchmark_id, **kwargs)
            save_result(settings, report)
            with _job_lock:
                _current_job = {"id": benchmark_id, "status": "done", "started_at": _current_job["started_at"]}
        except Exception as exc:
            with _job_lock:
                _current_job = {
                    "id": benchmark_id, "status": "error", "error": str(exc),
                    "started_at": _current_job["started_at"] if _current_job else time.time(),
                }

    threading.Thread(target=_run, daemon=True).start()
    return {"id": benchmark_id, "status": "started"}

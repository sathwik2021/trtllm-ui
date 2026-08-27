"""
build_sweep.py -- Automated deployment-CONFIGURATION sweep + benchmark harness.

READ THIS BEFORE RUNNING.

SCOPE, STATED PLAINLY: there is no plurality of "TensorRT engine build
types" available in this setup. `trtllm-build` is CONFIRMED ABSENT from
the current image (nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22 -- see
this project's own Phase 0 diagnostic). What this script actually sweeps
is DEPLOYMENT CONFIGURATION of the one confirmed-working path
(trtllm-serve, PyTorch backend, against the HF checkpoint directly) --
CLI flags and --config YAML options, not different engine "builds".

MEASURED NOISE FLOOR, NOT ASSUMED: this project has already measured
50%+ throughput swings between back-to-back runs with ZERO config
changes (root cause undetermined -- GPU thermal/power throttling was
directly ruled out via live nvidia-smi monitoring during a run). Because
of this, every variant below runs REPEATS times and is ranked by MEDIAN
plus a rough statistical comparison against baseline, never a single
sample. Even so, treat the ranking as directional, not conclusive,
unless a gap is clearly larger than that ~50% noise floor.

VARIANT LIST PROVENANCE: this project's own context-dump prompt (see
other_ai_prompt.md) was sent to 4 independent AI assistants for
suggestions. Their responses were cross-checked against THIS PROJECT'S
OWN PRIMARY EVIDENCE (the real captured `trtllm-serve serve --help`
output and the real captured `LLM Args:` log dump from an actual
deployment) before being included here -- not accepted at face value.
Specifically:
  - "max_num_seqs" (suggested by one source) does NOT appear anywhere in
    this build's real flags or LLM Args dump -- looks like a vLLM-ism
    hallucination, NOT included.
  - "stream_interval" (suggested by one source, real setting) is
    excluded on a different, mechanical ground: this project's own
    benchmark harness (benchmark_manager.py) sets "stream": False on
    every request, so stream_interval's entire mechanism (streaming-
    chunk cadence) cannot matter for what this benchmark measures.
  - "attn_backend" and "scheduler_config.capacity_scheduler_policy" were
    each flagged as uncertain/unavailable by some sources and confirmed
    real by others -- CONFIRMED REAL here via this project's own actual
    LLM Args dump (attn_backend='TRTLLM', SchedulerConfig(capacity_
    scheduler_policy=<CapacitySchedulerPolicy.GUARANTEED_NO_EVICT...)).
    Neither is a top-level CLI flag; both only reachable via the
    --config/extra_llm_api_options YAML mechanism.

WALL-CLOCK COST: each variant = stop existing deployment + fresh
`docker run` + wait for boot (~2-3 min observed for this 1.5B model,
timeout set generously above that) + REPEATS benchmark runs. With the
default variant list this will run for HOURS, unattended. That is the
explicit point (zero intervention except checking the final result) --
just don't expect it to be quick, and don't need the GPU/machine for
anything else while it runs.

SAFETY: every network call and every wait loop has a timeout. A variant
that fails to deploy, never becomes ready, or errors during benchmarking
is recorded as failed and the sweep moves on to the next one -- it will
never hang indefinitely on a broken config.

PREREQUISITE: the main app (`python run.py`) must already be running --
this script talks to it over HTTP (same as your browser does), reusing
all of its existing preflight/health-check/self-healing logic rather
than reimplementing or duplicating deployment state tracking.

USAGE:
  python -m backend.build_sweep                  # run the full sweep
  python -m backend.build_sweep --dry-run         # print the plan, do
                                                   # nothing else (sanity
                                                   # check before committing
                                                   # hours to a real run)
  python -m backend.build_sweep --interleave      # round-robin across
                                                   # variants instead of
                                                   # finishing one before
                                                   # the next -- costs
                                                   # ~REPEATS-times more
                                                   # redeploy overhead,
                                                   # see run_sweep()'s
                                                   # docstring for why
                                                   # this isn't the
                                                   # default here
  python -m backend.build_sweep --report-only     # print the ranking
                                                   # from the most recent
                                                   # saved sweep, run
                                                   # nothing
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

APP_BASE = "http://127.0.0.1:8420"
MODEL_NAME = "Qwen2.5-1.5B-Instruct"   # change if sweeping a different model
PORT = 8000

DEPLOY_BOOT_TIMEOUT_S = 400            # generous vs the ~160-260s worst case actually observed
DEPLOY_POLL_INTERVAL_S = 5

# Chosen to sit past the confirmed real batching-scaling knee (~concurrency
# 250+ range) without being so extreme that a single bad variant's queueing
# blowup burns 10+ minutes per repeat -- request_count kept >= concurrency
# so this never accidentally becomes the "burst, not sustained load"
# measurement gap found earlier in this project's own testing.
BENCHMARK_REQUEST_COUNT = 200
BENCHMARK_CONCURRENCY = 100
BENCHMARK_MAX_TOKENS = 256
PER_BENCHMARK_TIMEOUT_S = 600

REPEATS = 4  # up from an earlier draft's 2 -- still short of the 8-10
             # "useful" threshold multiple outside sources converged on
             # given the measured ~50% noise floor, but each repeat costs
             # real minutes. Raise via --repeats if you have hours to spare.

# ---- variant definitions ----
# See the module docstring's "VARIANT LIST PROVENANCE" section for how
# and why each of these was included or excluded.
VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline_defaults", "config": {}},
    {"name": "mem_fraction_0.95", "config": {"free_gpu_memory_fraction": 0.95}},
    {"name": "chunked_prefill", "config": {"extra_flags": {"enable_chunked_prefill": True}}},
    {
        "name": "mem_0.95_plus_chunked_prefill",
        "config": {"free_gpu_memory_fraction": 0.95, "extra_flags": {"enable_chunked_prefill": True}},
    },
    {
        "name": "cuda_graph_padding",
        "config": {"extra_llm_api_options": {"cuda_graph_config": {"enable_padding": True}}},
        # UNCONFIRMED net effect -- inconclusive in earlier single-shot
        # testing due to the noise floor. Kept because the mechanism is
        # real and cheap to re-test properly with repeats this time.
    },
    {
        "name": "cuda_graph_padding_batch256",
        "config": {"extra_llm_api_options": {"cuda_graph_config": {"enable_padding": True, "max_batch_size": 256}}},
        # Real, confirmed default graph ceiling is 128 (from this
        # project's own logs: "Creating CUDA graph instances for 34
        # batch sizes" up to 128) -- real operating point (concurrency
        # 250-600) exceeds that. CAUTION: already ~5.4-5.7GB / 6GB VRAM
        # at saturation -- capturing more/larger graphs costs memory.
        # Watch docker logs for OOM on this one, not just throughput.
    },
    {
        "name": "attn_backend_flashinfer",
        "config": {"extra_llm_api_options": {"attn_backend": "FLASHINFER"}},
        # EXPERIMENTAL. Confirmed real field (attn_backend='TRTLLM' is
        # the actual confirmed default). Only 1 of 4 outside sources
        # flagged this at all. FlashInfer is documented as supporting
        # SM86 generally, but there's no confirmed evidence it's built
        # into this exact image or beats the default backend for this
        # model -- genuinely unknown until tested. A boot failure here
        # is information (backend unavailable in this image), not
        # necessarily a bug in this script.
    },
    {
        "name": "scheduler_max_utilization",
        "config": {"extra_llm_api_options": {"scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"}}},
        # Confirmed real (default is GUARANTEED_NO_EVICT). Expected to
        # matter mainly under KV-cache pressure; this deployment is
        # compute-saturated before VRAM-saturated, so most sources
        # expect little or no gain, possibly worse tail latency if it
        # starts evicting/pausing requests under pressure.
    },
    {"name": "postprocess_workers_4", "config": {"extra_flags": {"num_postprocess_workers": 4}}},
    {"name": "no_telemetry", "config": {"extra_flags": {"no-telemetry": True}}},
    {
        "name": "kitchen_sink",
        "config": {
            "free_gpu_memory_fraction": 0.95,
            "extra_flags": {"enable_chunked_prefill": True, "num_postprocess_workers": 4, "no-telemetry": True},
        },
    },
]


def _http_json(method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{APP_BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _deployment_id_for(model_name: str) -> str:
    # Mirrors deployment_manager._slugify's behavior for the common case
    # (dots/underscores -> hyphens, lowercased). If your model name slugs
    # differently, override with --deployment-id.
    return model_name.lower().replace(".", "-").replace("_", "-")


def stop_existing(deployment_id: str) -> None:
    try:
        _http_json("POST", f"/api/deployments/{deployment_id}/stop", timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    time.sleep(3)  # let Docker actually release the port/name before redeploying


def deploy_variant(model_name: str, port: int, variant_config: dict) -> str:
    body = {"model_name": model_name, "port": port, **variant_config}
    result = _http_json("POST", "/api/deployments", body, timeout=60)
    return result["deployment_id"]


def wait_until_ready(deployment_id: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = _http_json("GET", f"/api/deployments/{deployment_id}")
        except Exception:
            status = None
        if status and status.get("status") == "ready":
            return True
        if status and status.get("status") == "error":
            return False
        time.sleep(DEPLOY_POLL_INTERVAL_S)
    return False


def _record_run(report: dict[str, Any]) -> dict[str, Any]:
    """Safely extract (throughput, p50) from a completed benchmark report
    and print a one-line summary -- never assumes p50 is a number.

    Real crash this fixes: a benchmark job can complete successfully at
    the job level (no top-level "error" key -- it started, ran, and
    finished) while EVERY individual request inside it failed. That's a
    real, confirmed benchmark_manager.py behavior (see its own
    test_zero_successful_requests_does_not_crash_on_throughput), not an
    edge case invented here: throughput_tokens_per_s comes back as a real
    0.0 (0 tokens / real wall time), but latency_s.p50 is None (no
    successful request latencies exist to take a percentile of). This
    project hit exactly that case in real, unattended use -- one repeat
    of one variant had zero successful requests, and the old code's bare
    f"{p50:.1f}" formatting crashed on None, aborting the ENTIRE REST OF
    THE SWEEP and losing every remaining variant's data. That failure
    mode is worse than the bug itself, given the whole point of this
    script is to survive bad variants unattended.
    """
    throughput = report.get("throughput_tokens_per_s")
    latency = report.get("latency_s") or {}
    p50 = latency.get("p50")
    ok_count = report.get("requests_ok")
    failed_count = report.get("requests_failed")
    if p50 is None:
        t_str = f"{throughput:.1f}" if throughput is not None else "unknown"
        print(f"    completed with ZERO successful requests (throughput={t_str} tok/s, ok={ok_count}, failed={failed_count})")
    else:
        print(f"    throughput={throughput:.1f} tok/s, p50={p50:.1f}s (ok={ok_count}, failed={failed_count})")
    return {"id": report.get("id"), "throughput": throughput, "p50": p50, "ok_count": ok_count, "failed_count": failed_count}


def run_one_benchmark(deployment_id: str) -> dict[str, Any]:
    body = {
        "deployment_id": deployment_id,
        "request_count": BENCHMARK_REQUEST_COUNT,
        "concurrency": BENCHMARK_CONCURRENCY,
        "max_tokens": BENCHMARK_MAX_TOKENS,
    }
    try:
        start = _http_json("POST", "/api/benchmarks", body, timeout=30)
    except Exception as exc:
        return {"error": f"failed to start benchmark: {exc}"}

    benchmark_id = start["id"]
    deadline = time.time() + PER_BENCHMARK_TIMEOUT_S
    while time.time() < deadline:
        try:
            job = _http_json("GET", "/api/benchmarks/status")
        except Exception as exc:
            return {"error": f"lost contact polling benchmark status: {exc}"}
        if job.get("status") == "done":
            return _http_json("GET", f"/api/benchmarks/{benchmark_id}")
        if job.get("status") == "error":
            return {"error": job.get("error", "unknown benchmark error")}
        time.sleep(5)
    return {"error": "benchmark timed out"}


def _deploy_and_wait(deployment_id: str, variant: dict) -> bool:
    stop_existing(deployment_id)
    try:
        deploy_variant(MODEL_NAME, PORT, variant["config"])
    except Exception as exc:
        print(f"  deploy failed: {exc}")
        return False
    if not wait_until_ready(deployment_id, DEPLOY_BOOT_TIMEOUT_S):
        print("  never became ready within timeout")
        return False
    return True


def run_single_variant(deployment_id: str, variant: dict, repeats: int) -> dict[str, Any]:
    """Deploy ONE variant and run `repeats` benchmarks against it, right
    now, as whatever position in the session this happens to be. Exists
    specifically to test the session-order hypothesis raised by a real
    sweep result: every variant tested 3rd-or-later in that run clustered
    tightly around ~2450-2490 tok/s regardless of its actual config,
    while the 1st and 2nd variants tested (including baseline) sat
    ~30% lower and never recovered within their own 4 repeats. If
    redeploying baseline_defaults again, later in a session, also lands
    in that same higher band, that's real evidence the sweep's ranking
    was confounded by deploy order, not by the config differences it
    was trying to measure.
    """
    name = variant["name"]
    print(f"\n=== single retest: {name} ===", flush=True)
    if not _deploy_and_wait(deployment_id, variant):
        return {"variant": name, "config": variant["config"], "status": "boot_failed"}

    throughputs: list[float] = []
    p50s: list[float] = []
    raw_runs: list[dict[str, Any]] = []
    for i in range(repeats):
        print(f"  benchmark repeat {i + 1}/{repeats}...", flush=True)
        report = run_one_benchmark(deployment_id)
        if "error" in report:
            print(f"    failed: {report['error']}")
            raw_runs.append({"error": report["error"]})
            continue
        run_info = _record_run(report)
        raw_runs.append(run_info)
        if run_info["throughput"] is not None:
            throughputs.append(run_info["throughput"])
        if run_info["p50"] is not None:
            p50s.append(run_info["p50"])

    return {
        "variant": name,
        "config": variant["config"],
        "status": "ok" if throughputs else "all_benchmarks_failed",
        "throughput_median": statistics.median(throughputs) if throughputs else None,
        "throughput_all_runs": throughputs,
        "p50_median": statistics.median(p50s) if p50s else None,
        "raw_runs": raw_runs,
        "ran_at": time.time(),
        "ran_as": "single_retest",
    }


def _rank_with_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def mannwhitney_p(a: list[float], b: list[float]) -> float | None:
    """Two-sided Mann-Whitney U test, normal approximation with tie
    correction, no scipy dependency. With only a handful of repeats per
    side (REPEATS=4 by default here) this is a rough approximation, not
    a rigorous p-value -- report it as weak directional evidence on top
    of the raw numbers, not as a definitive verdict. Returns None if
    either sample is empty or the combined sample size is too small to
    say anything.
    """
    if not a or not b:
        return None
    combined = a + b
    n1, n2 = len(a), len(b)
    n_total = n1 + n2
    if n_total <= 1:
        return None
    ranks = _rank_with_ties(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    counts = Counter(combined)
    tie_term = sum(t ** 3 - t for t in counts.values())
    variance = (n1 * n2 / 12) * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
    if variance <= 0:
        return 1.0
    sigma = variance ** 0.5
    z = (u - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return max(0.0, min(1.0, p))


def run_sweep(deployment_id: str, interleave: bool, repeats: int) -> dict[str, Any]:
    """Two execution modes -- pick deliberately, see module docstring.

    interleave=False (default): deploy each variant ONCE, run `repeats`
    benchmarks back-to-back against that same live deployment. Cheaper
    (one deploy cycle per variant). Vulnerable to drift on an HOURS
    timescale (e.g. thermal soak across the whole sweep) landing
    unevenly across variants run at different points in the sweep.

    interleave=True: redeploy before EVERY single benchmark run, cycling
    through variants in round-robin order. ~`repeats`-times more
    redeploy overhead. Protects against hours-scale drift, but this
    project's own measured noise (50%+ swings between back-to-back runs
    of the IDENTICAL config, seconds apart) suggests the dominant noise
    source here operates on a faster timescale than interleaving is
    designed to fix -- offered, not assumed correct for this pattern.
    """
    variant_data: dict[str, dict[str, Any]] = {
        v["name"]: {"variant": v, "throughputs": [], "p50s": [], "raw_runs": [], "status": "pending"}
        for v in VARIANTS
    }

    if not interleave:
        for variant in VARIANTS:
            name = variant["name"]
            print(f"\n=== {name} ===", flush=True)
            d = variant_data[name]
            if not _deploy_and_wait(deployment_id, variant):
                d["status"] = "boot_failed"
                continue
            for i in range(repeats):
                print(f"  benchmark repeat {i + 1}/{repeats}...", flush=True)
                report = run_one_benchmark(deployment_id)
                if "error" in report:
                    print(f"    failed: {report['error']}")
                    d["raw_runs"].append({"error": report["error"]})
                    continue
                run_info = _record_run(report)
                d["raw_runs"].append(run_info)
                if run_info["throughput"] is not None:
                    d["throughputs"].append(run_info["throughput"])
                if run_info["p50"] is not None:
                    d["p50s"].append(run_info["p50"])
            d["status"] = "ok" if d["throughputs"] else "all_benchmarks_failed"
    else:
        for round_num in range(repeats):
            for variant in VARIANTS:
                name = variant["name"]
                print(f"\n=== round {round_num + 1}/{repeats} -- {name} ===", flush=True)
                d = variant_data[name]
                if not _deploy_and_wait(deployment_id, variant):
                    d["status"] = "boot_failed"
                    continue
                report = run_one_benchmark(deployment_id)
                if "error" in report:
                    print(f"  failed: {report['error']}")
                    d["raw_runs"].append({"error": report["error"]})
                    continue
                run_info = _record_run(report)
                d["raw_runs"].append(run_info)
                if run_info["throughput"] is not None:
                    d["throughputs"].append(run_info["throughput"])
                if run_info["p50"] is not None:
                    d["p50s"].append(run_info["p50"])
        for d in variant_data.values():
            if d["status"] != "boot_failed":
                d["status"] = "ok" if d["throughputs"] else "all_benchmarks_failed"

    results = []
    for name, d in variant_data.items():
        results.append({
            "variant": name,
            "config": d["variant"]["config"],
            "status": d["status"],
            "throughput_median": statistics.median(d["throughputs"]) if d["throughputs"] else None,
            "throughput_all_runs": d["throughputs"],
            "p50_median": statistics.median(d["p50s"]) if d["p50s"] else None,
            "raw_runs": d["raw_runs"],
        })

    return {
        "ran_at": time.time(),
        "model": MODEL_NAME,
        "concurrency": BENCHMARK_CONCURRENCY,
        "request_count": BENCHMARK_REQUEST_COUNT,
        "repeats": repeats,
        "interleaved": interleave,
        "results": results,
    }


def print_ranking(sweep: dict[str, Any]) -> None:
    ok = [r for r in sweep["results"] if r["status"] == "ok"]
    ok.sort(key=lambda r: r["throughput_median"] or 0, reverse=True)

    baseline = next((r for r in sweep["results"] if r["variant"] == "baseline_defaults"), None)
    baseline_runs = baseline["throughput_all_runs"] if baseline else []

    print("\n=== RANKING (median throughput across repeats, higher is better) ===")
    for r in ok:
        runs = ", ".join(f"{v:.0f}" for v in r["throughput_all_runs"])
        p_str = ""
        if r["variant"] != "baseline_defaults" and baseline_runs:
            p = mannwhitney_p(r["throughput_all_runs"], baseline_runs)
            if p is not None:
                sig = "significant-ish" if p < 0.05 else "not distinguishable from noise"
                p_str = f"  vs baseline: p~{p:.3f} ({sig})"
        print(f"{r['throughput_median']:.1f} tok/s  (p50 {r['p50_median']:.1f}s)  -- {r['variant']}   [runs: {runs}]{p_str}")

    failed = [r for r in sweep["results"] if r["status"] != "ok"]
    if failed:
        print("\n=== FAILED / SKIPPED ===")
        for r in failed:
            print(f"{r['variant']}: {r['status']}" + (f" ({r['error']})" if "error" in r else ""))

    print(
        "\nCaveats, read before trusting anything above:\n"
        "1. This project measured 50%+ run-to-run throughput noise with NO "
        "config change at all. A ranking gap smaller than that is not "
        "meaningful.\n"
        "2. The p-values are a normal-approximation Mann-Whitney U test "
        "with only a handful of repeats per side -- treat as weak "
        "directional evidence, not a rigorous statistical result, "
        f"especially at repeats={sweep.get('repeats')}.\n"
        f"3. interleaved={sweep.get('interleaved')} -- if False, each "
        "variant's repeats ran back-to-back rather than round-robin "
        "across variants; a slow drift over the sweep's total runtime "
        "could still confound the ranking."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="Print the ranking from the most recent saved sweep; run nothing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the variant plan and exit; deploys/benchmarks nothing.")
    parser.add_argument("--deployment-id", default=None, help="Override the auto-derived deployment id if it doesn't match your model's actual slug.")
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"Benchmark repeats per variant (default {REPEATS}).")
    parser.add_argument(
        "--interleave", action="store_true",
        help="Round-robin across variants, redeploying before every single benchmark "
             "run, instead of finishing one variant before the next. Costs ~repeats-x "
             "more redeploy overhead. See run_sweep()'s docstring for why this isn't "
             "the default given this project's specific measured noise pattern.",
    )
    parser.add_argument(
        "--variant", default=None, metavar="NAME",
        help="Deploy and benchmark ONE named variant right now, instead of running the "
             "full sweep. Useful for testing the session-order hypothesis (redeploy "
             "baseline_defaults again later in a session and see if it now lands in a "
             "different performance band than it did as the 1st deployment) or for "
             "quickly re-checking a single result without re-running everything. "
             "Valid names: " + ", ".join(v["name"] for v in VARIANTS),
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "sweep_results"
    out_dir.mkdir(exist_ok=True)

    if args.report_only:
        files = sorted(out_dir.glob("*.json"))
        if not files:
            print("No sweep results found yet -- run a sweep first.")
            return
        print_ranking(json.loads(files[-1].read_text(encoding="utf-8")))
        return

    if args.variant:
        matches = [v for v in VARIANTS if v["name"] == args.variant]
        if not matches:
            print(f"Unknown variant '{args.variant}'. Valid names:")
            for v in VARIANTS:
                print(f"  - {v['name']}")
            return
        variant = matches[0]
        deployment_id = args.deployment_id or _deployment_id_for(MODEL_NAME)

        if args.dry_run:
            print(f"Would deploy and benchmark ONLY: {variant['name']}: {json.dumps(variant['config'])}")
            print(f"Deployment id: {deployment_id}, repeats: {args.repeats}")
            return

        print(f"Single-variant retest: {variant['name']}: {json.dumps(variant['config'])}")
        print(f"Deployment id: {deployment_id}, repeats: {args.repeats}")
        input("\nPress Enter to start (Ctrl+C to cancel)... ")

        result = run_single_variant(deployment_id, variant, args.repeats)
        out_path = out_dir / f"single_{variant['name']}_{int(time.time())}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved to {out_path}")
        if result["status"] == "ok":
            print(f"\nMedian throughput: {result['throughput_median']:.1f} tok/s (runs: {result['throughput_all_runs']})")
            print(
                "\nCompare this against this variant's entry in your last full sweep's "
                "JSON. If the number is dramatically different despite an identical "
                "config, that's evidence of session-order/drift confounding rather "
                "than a real config effect -- see the note in the previous sweep's "
                "printed ranking."
            )
        else:
            print(f"\nStatus: {result['status']}")
        return

    if args.dry_run:
        print(__doc__)
        print(f"\n{len(VARIANTS)} variant(s), {args.repeats} repeat(s) each = {len(VARIANTS) * args.repeats} total benchmark runs.")
        print(f"Interleaved: {args.interleave} ({'redeploy every single run' if args.interleave else 'redeploy once per variant'})")
        for v in VARIANTS:
            print(f"  - {v['name']}: {json.dumps(v['config'])}")
        return

    deployment_id = args.deployment_id or _deployment_id_for(MODEL_NAME)
    print(__doc__)
    print(f"Deployment id resolved to: {deployment_id}")
    print(f"Interleaved: {args.interleave}, repeats: {args.repeats}")
    input("\nPress Enter to start the sweep (Ctrl+C to cancel)... ")

    sweep = run_sweep(deployment_id, interleave=args.interleave, repeats=args.repeats)
    out_path = out_dir / f"sweep_{int(sweep['ran_at'])}.json"
    out_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")
    print_ranking(sweep)


if __name__ == "__main__":
    main()

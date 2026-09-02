from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from .settings import AppSettings, to_host_path


_probe_lock = threading.Lock()


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        return {
            "argv": cmd,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": p.returncode,
        }
    except Exception as exc:
        return {"argv": cmd, "stdout": "", "stderr": str(exc), "returncode": -1}


def _parse_flags(text: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        for match in re.finditer(r"(--[A-Za-z0-9_][A-Za-z0-9_-]*)", line):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            remainder = line[match.start():].strip()
            flags.append({
                "name": name,
                "takes_value": bool(re.search(r"--[A-Za-z0-9_][A-Za-z0-9_-]*[ =]+(?:[A-Za-z<\[]|[A-Z_][A-Z0-9_-]*)", remainder)),
                "help": remainder,
            })
    return flags


def _enum_choices(serve_flags: list[dict[str, Any]] | None, *flag_names: str) -> list[str] | None:
    """Extract the real `[choice1|choice2|...]` enum values for a CLI flag
    from its parsed --help line, e.g. `--kv_cache_dtype [auto|fp8|nvfp4]`.
    Returns None if the flag wasn't found in this probe (unprobed / flag
    renamed / removed in a newer image) -- callers must treat that as
    "unknown", not as "no choices" or a static guess.
    """
    if not serve_flags:
        return None
    for flag in serve_flags:
        if flag["name"] in flag_names:
            m = re.search(r"\[([a-zA-Z0-9_]+(?:\|[a-zA-Z0-9_]+)+)\]", flag["help"])
            if m:
                return m.group(1).split("|")
    return None


def _image_tag(image: str) -> str:
    return image.replace("/", "_").replace(":", "_").replace("@", "_")


def _parse_gpus(csv_text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap
    --format=csv` output. Returns [] if nvidia-smi failed or produced nothing --
    callers must treat an empty list as "unknown", never assume 1 GPU exists.
    """
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines or "name" not in lines[0].lower():
        return []
    gpus = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, mem_total, mem_free, driver, compute_cap = parts[:5]
        try:
            total_mb = int(re.sub(r"[^0-9]", "", mem_total) or 0)
        except ValueError:
            total_mb = None
        try:
            free_mb = int(re.sub(r"[^0-9]", "", mem_free) or 0)
        except ValueError:
            free_mb = None
        try:
            cc = float(compute_cap)
        except ValueError:
            cc = None
        gpus.append({
            "name": name,
            "vram_total_mb": total_mb,
            "vram_free_mb": free_mb,
            "driver_version": driver,
            "compute_capability": cc,
        })
    return gpus


def _precision_capabilities(compute_cap: float | None) -> dict[str, Any]:
    """Derive precision/quantization capability from compute capability (SM version).
    This is a HARDWARE ceiling, not a software confirmation -- an entry marked
    supported=True here still needs runtime testing against this exact TRT-LLM
    build before being trusted; entries marked supported=False are a hard
    hardware exclusion (no image/version/config change will enable them) and
    should be omitted from UI option lists entirely, not shown disabled.

    Thresholds (Compute Capability / SM version):
      FP16   -- supported on all CUDA-capable GPUs this app targets
      BF16   -- Ampere (SM80) and later have BF16 tensor core paths
      INT8   -- Turing (SM75) and later have INT8 tensor core paths
      FP8    -- Ada Lovelace (SM89) / Hopper (SM90) and later ONLY
      NVFP4  -- Blackwell (SM100) and later ONLY
    """
    if compute_cap is None:
        return {
            "fp16": {"supported": None, "confidence": "unknown", "note": "compute capability not detected"},
            "bf16": {"supported": None, "confidence": "unknown", "note": "compute capability not detected"},
            "int8": {"supported": None, "confidence": "unknown", "note": "compute capability not detected"},
            "fp8": {"supported": None, "confidence": "unknown", "note": "compute capability not detected"},
            "nvfp4": {"supported": None, "confidence": "unknown", "note": "compute capability not detected"},
        }
    return {
        "fp16": {"supported": True, "confidence": "confirmed", "note": "baseline precision, already the working setup"},
        "bf16": {
            "supported": compute_cap >= 8.0,
            "confidence": "hardware_only" if compute_cap >= 8.0 else "confirmed",
            "note": "Ampere+ has BF16 tensor cores; needs testing against this TRT-LLM build specifically" if compute_cap >= 8.0 else "no BF16 tensor core path on this GPU",
        },
        "int8": {
            "supported": compute_cap >= 7.5,
            "confidence": "hardware_only" if compute_cap >= 7.5 else "confirmed",
            "note": "hardware capable; software path (calibration/quantization tooling) unconfirmed on this image" if compute_cap >= 7.5 else "no INT8 tensor core path on this GPU",
        },
        "fp8": {
            "supported": compute_cap >= 8.9,
            "confidence": "confirmed",
            "note": "requires Ada Lovelace (SM89) or Hopper (SM90)+; hard hardware exclusion below that" if compute_cap < 8.9 else "hardware capable; needs testing against this TRT-LLM build",
        },
        "nvfp4": {
            "supported": compute_cap >= 10.0,
            "confidence": "confirmed",
            "note": "requires Blackwell (SM100)+; hard hardware exclusion below that" if compute_cap < 10.0 else "hardware capable; needs testing against this TRT-LLM build",
        },
    }


def _kv_cache_dtype_options(precision: dict[str, Any], serve_flags: list[dict[str, Any]] | None = None) -> list[str]:
    """Which --kv_cache_dtype values are worth offering in the UI.

    Confirmed real choices come from the probed `serve --help` text (a
    build/version can rename or add/drop values -- e.g. this project's
    confirmed 1.3.0rc22 build only accepts auto|fp8|nvfp4, NOT the
    generic int8/fp16/bf16 set an earlier version of this function
    assumed without checking). If serve_flags aren't available yet
    (not probed), falls back to a conservative static guess -- always
    prefer the real parsed choices when present.

    Either way, non-"auto" choices are still gated on hardware capability:
    fp8/nvfp4 in the CLI's own enum doesn't mean THIS gpu can execute them.
    """
    real_choices = _enum_choices(serve_flags, "--kv_cache_dtype")
    candidates = real_choices if real_choices is not None else ["auto", "fp8", "nvfp4"]

    options = []
    for c in candidates:
        c_norm = c.lower()
        if c_norm == "auto":
            options.append(c)
        elif c_norm in ("fp8",) and precision.get("fp8", {}).get("supported"):
            options.append(c)
        elif c_norm in ("nvfp4", "fp4") and precision.get("nvfp4", {}).get("supported"):
            options.append(c)
        elif c_norm == "int8" and precision.get("int8", {}).get("supported"):
            # Not present in the confirmed 1.3.0rc22 enum, kept only in case
            # a different image version's --help genuinely offers it.
            options.append(c)
    return options


def run_probe(settings: AppSettings) -> dict[str, Any]:
    with _probe_lock:
        image = settings.docker_image
        commands = [
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap", "--format=csv"],
            ["docker", "system", "df", "--format", "json"],
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            [
                "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
                "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                image, "trtllm-serve", "serve", "--help",
            ],
            ["docker", "run", "--rm", "--gpus", "all", image, "trtllm-serve", "--help"],
        ]

        results = []
        for index, cmd in enumerate(commands):
            result = _run(cmd)
            if index == 1 and result["returncode"] != 0:
                result = _run(["docker", "system", "df"])
            results.append(result)

        gpus = _parse_gpus(results[0]["stdout"])
        gpu_count = len(gpus)
        # Compute capability may legitimately differ across GPUs on a
        # multi-GPU box; precision capability is computed per-GPU and the
        # UI-facing "precision" summary uses the MINIMUM across all
        # detected GPUs, since a mixed-precision UI offer that only some
        # GPUs can execute is worse than an overly conservative one.
        per_gpu_precision = [_precision_capabilities(g["compute_capability"]) for g in gpus]
        if per_gpu_precision:
            precision = {}
            for key in per_gpu_precision[0]:
                supported_vals = [p[key]["supported"] for p in per_gpu_precision]
                precision[key] = per_gpu_precision[0][key] if len(per_gpu_precision) == 1 else {
                    "supported": all(bool(v) for v in supported_vals) if None not in supported_vals else None,
                    "confidence": per_gpu_precision[0][key]["confidence"],
                    "note": per_gpu_precision[0][key]["note"],
                }
        else:
            precision = _precision_capabilities(None)

        manifest = {
            "probed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image": image,
            "image_tag": _image_tag(image),
            "commands": results,
            "parsed": {
                "serve_flags": (serve_flags := _parse_flags(
                    results[3]["stdout"] + "\n" + results[3]["stderr"]
                )),
                "gpu": {
                    "count": gpu_count,
                    "gpus": gpus,
                    # None (not 1) when nvidia-smi produced nothing -- callers
                    # must not silently assume single-GPU on a failed probe.
                },
                "precision": precision,
                "kv_cache_dtype_options": _kv_cache_dtype_options(precision, serve_flags),
                # Parallelism dimensions are only meaningful with >1 GPU.
                # max=1 (or unknown if gpu_count is 0/undetected) means the
                # UI should hide the corresponding selector, not just default it to 1.
                "parallel_max": gpu_count if gpu_count > 0 else None,
            },
        }
        settings.materialize_dirs()
        out = to_host_path(settings.data_dir) / f"capability_manifest_{_image_tag(image)}.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def cached_manifest(settings: AppSettings) -> dict[str, Any] | None:
    path = to_host_path(settings.data_dir) / f"capability_manifest_{_image_tag(settings.docker_image)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

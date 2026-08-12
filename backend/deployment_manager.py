from __future__ import annotations

import json
import re
import shlex
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from pydantic import BaseModel, Field

from . import gpu_monitor, vram_estimator
from .capability_probe import cached_manifest
from .model_manager import get_model
from .settings import AppSettings


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return slug[:40] or "deployment"


class DeploymentConfig(BaseModel):
    # Optional stable identity for this deployment. If omitted, derived from
    # model_name. Two deploy calls with the same resulting deployment_id map
    # to the SAME container name, which is what makes create/start/reuse
    # (see start_deployment) possible instead of minting a new container
    # every time.
    name: str | None = None
    model_name: str
    backend: str = "pytorch"
    host: str = "0.0.0.0"
    port: int = 8000
    served_model_name: str | None = None
    max_batch_size: int | None = None
    max_seq_len: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    moe_expert_parallel_size: int = 1
    gpus_per_node: int | None = None
    kv_cache_dtype: str | None = None
    free_gpu_memory_fraction: float | None = None
    trust_remote_code: bool = False
    custom_module_dirs: str | None = None
    unsafe_ack: str | None = None
    extra_flags: dict[str, Any] = Field(default_factory=dict)
    # Informational only -- NOT a trtllm-serve CLI flag (no confirmed
    # --max_output_len equivalent exists in this build's serve --help
    # output). Used purely as an input to the VRAM pre-flight estimate
    # (see vram_estimator.estimate); build_command never emits it.
    max_output_tokens: int | None = None


_deployments: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _docker_inspect(name: str) -> dict[str, Any] | None:
    p = subprocess.run(
        ["docker", "inspect", name],
        capture_output=True, text=True, check=False
    )
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)[0]
    except Exception:
        return None


def _extract_port(info: dict[str, Any] | None) -> int | None:
    """Recover the published host port from `docker inspect` output.

    Fixes the known reconcile() gap where rediscovered containers had
    port set to None, silently disabling port-collision checks after a
    restart.
    """
    if not info:
        return None
    try:
        ports = info.get("NetworkSettings", {}).get("Ports") or {}
        for _container_port, bindings in ports.items():
            if not bindings:
                continue
            host_port = bindings[0].get("HostPort")
            if host_port:
                return int(host_port)
    except Exception:
        pass
    # Fallback: parse "--port <N>" out of the container's launch command.
    cmd = (info.get("Config", {}) or {}).get("Cmd") or []
    for i, arg in enumerate(cmd):
        if arg == "--port" and i + 1 < len(cmd):
            try:
                return int(cmd[i + 1])
            except ValueError:
                pass
    return None


def _reconstruct_config(info: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort reconstruction of a deployment's config from
    `docker inspect`, for containers discovered after an app restart
    rather than started by this process. This is NOT a full
    DeploymentConfig -- only what's actually recoverable from the
    container's launch args. Callers should treat this as informational,
    not authoritative.
    """
    if not info:
        return {}
    cmd = (info.get("Config", {}) or {}).get("Cmd") or []
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                parsed[key] = cmd[i + 1]
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            i += 1
    return {
        "backend": parsed.get("backend"),
        "served_model_name": parsed.get("served_model_name"),
        "image": info.get("Config", {}).get("Image"),
        "reconstructed": True,
        "reconstructed_note": "Best-effort, parsed from docker inspect Config.Cmd -- not the original request payload.",
    }


def _port_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _flag_name(key: str) -> str:
    return key if key.startswith("--") else f"--{key}"


def build_command(settings: AppSettings, config: DeploymentConfig, deployment_id: str | None = None) -> tuple[list[str], list[str]]:
    get_model(settings, config.model_name)
    deployment_id = deployment_id or _slugify(config.name or config.model_name)
    container_name = f"trtllm-ui-{deployment_id}"

    if not (1024 <= config.port <= 65535):
        raise ValueError("port must be between 1024 and 65535")
    with _lock:
        # Exclude this same deployment_id's own (possibly stale) record from
        # the collision check -- e.g. a previous container with this name was
        # removed outside the app, but the in-memory record wasn't cleared.
        occupied = {did: d["port"] for did, d in _deployments.items() if did != deployment_id}
    if config.port in occupied.values() or not _port_free(config.port):
        raise ValueError(f"port {config.port} is already in use")

    if config.tensor_parallel_size < 1 or config.pipeline_parallel_size < 1 or config.context_parallel_size < 1 or config.moe_expert_parallel_size < 1:
        raise ValueError("parallelism values must be >= 1")

    if (config.trust_remote_code or config.custom_module_dirs) and config.unsafe_ack != "ENABLE UNSAFE":
        raise ValueError('Unsafe flags require unsafe_ack == "ENABLE UNSAFE"')

    manifest_for_gate = cached_manifest(settings)
    if manifest_for_gate is not None and config.kv_cache_dtype:
        kv_options = manifest_for_gate.get("parsed", {}).get("kv_cache_dtype_options")
        if kv_options is not None and config.kv_cache_dtype.lower() not in [o.lower() for o in kv_options]:
            raise ValueError(
                f"kv_cache_dtype '{config.kv_cache_dtype}' is not supported by the detected GPU/image "
                f"(supported: {kv_options}). This is a hardware/build capability gate, not a typo check."
            )

    model = get_model(settings, config.model_name)
    cmd = [
        "docker", "run", "-d",
        "--restart", "unless-stopped",
        "--name", container_name,
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-p", f"127.0.0.1:{config.port}:{config.port}",
        "-v", f"{model['host_path']}:{model['container_path']}:ro",
        settings.docker_image,
        "trtllm-serve", "serve",
        model["container_path"],
        "--backend", config.backend,
        "--host", config.host,
        "--port", str(config.port),
        "--served_model_name", config.served_model_name or config.model_name,
    ]

    optional = {
        "max_batch_size": config.max_batch_size,
        "max_seq_len": config.max_seq_len,
        "tensor_parallel_size": config.tensor_parallel_size if config.tensor_parallel_size != 1 else None,
        "pipeline_parallel_size": config.pipeline_parallel_size if config.pipeline_parallel_size != 1 else None,
        "context_parallel_size": config.context_parallel_size if config.context_parallel_size != 1 else None,
        "moe_expert_parallel_size": config.moe_expert_parallel_size if config.moe_expert_parallel_size != 1 else None,
        "gpus_per_node": config.gpus_per_node,
        "kv_cache_dtype": config.kv_cache_dtype,
        "free_gpu_memory_fraction": config.free_gpu_memory_fraction,
    }
    for key, value in optional.items():
        if value is not None:
            cmd.extend([_flag_name(key), str(value)])

    if config.trust_remote_code:
        cmd.append("--trust_remote_code")
    if config.custom_module_dirs:
        cmd.extend(["--custom_module_dirs", config.custom_module_dirs])

    warnings = []
    manifest = cached_manifest(settings)
    known = {
        x["name"].lstrip("-")
        for x in (manifest or {}).get("parsed", {}).get("serve_flags", [])
    }
    for key, value in config.extra_flags.items():
        normalized = key.lstrip("-")
        if known and normalized not in known:
            warnings.append(f"Unknown capability flag: {key}")
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None and value != "":
            cmd.extend([flag, str(value)])

    return cmd, warnings


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _watch(deployment_id: str, port: int) -> None:
    deadline = time.time() + 90
    while time.time() < deadline:
        info = _docker_inspect(_deployments[deployment_id]["container_name"])
        if info is None:
            _deployments[deployment_id]["status"] = "error"
            _deployments[deployment_id]["reason"] = "container not found"
            return
        state = info.get("State", {})
        if state.get("Status") == "exited":
            _deployments[deployment_id]["status"] = "error"
            _deployments[deployment_id]["reason"] = "container exited during startup"
            return
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    _deployments[deployment_id]["status"] = "ready"
                    return
        except Exception:
            pass
        time.sleep(2)
    _deployments[deployment_id]["status"] = "error"
    _deployments[deployment_id]["reason"] = "startup timeout — check logs"


def create_container(settings: AppSettings, config: DeploymentConfig) -> dict[str, Any]:
    """Create a brand-new container (`docker run -d`). Only safe to call
    when no container with the derived name already exists -- callers
    should go through start_deployment(), which checks this first.
    """
    deployment_id = _slugify(config.name or config.model_name)
    cmd, warnings = build_command(settings, config, deployment_id)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "docker run failed").strip())
    record = {
        "deployment_id": deployment_id,
        "container_name": f"trtllm-ui-{deployment_id}",
        "config": config.model_dump(),
        "port": config.port,
        "status": "starting",
        "warnings": warnings,
        "command": cmd,
    }
    with _lock:
        _deployments[deployment_id] = record
    threading.Thread(target=_watch, args=(deployment_id, config.port), daemon=True).start()
    return record


def start_container(deployment_id: str, container_name: str, port: int | None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start an existing, stopped container (`docker start`), rather than
    creating a new one. Used when a container with the derived name
    already exists but isn't running.
    """
    p = subprocess.run(["docker", "start", container_name], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "docker start failed").strip())
    with _lock:
        record = _deployments.get(deployment_id)
        if record is None:
            record = {
                "deployment_id": deployment_id,
                "container_name": container_name,
                "config": config or {},
                "port": port,
                "status": "starting",
                "warnings": [],
                "command": ["docker", "start", container_name],
            }
            _deployments[deployment_id] = record
        else:
            record["status"] = "starting"
            if port is not None:
                record["port"] = port
    if port:
        threading.Thread(target=_watch, args=(deployment_id, port), daemon=True).start()
    return record


def start_deployment(settings: AppSettings, config: DeploymentConfig) -> dict[str, Any]:
    """Orchestrator implementing create / start / reuse:

    - no container by this (derived) name exists  -> create_container (docker run)
    - container exists, not running                -> start_container  (docker start)
    - container exists and already running          -> reuse: return its current
      status rather than attempting another `docker run`, which would fail
      with "name already in use" and previously wasn't handled at all.
    """
    deployment_id = _slugify(config.name or config.model_name)
    container_name = f"trtllm-ui-{deployment_id}"
    info = _docker_inspect(container_name)

    if info is None:
        return create_container(settings, config)

    state = info.get("State", {})
    if state.get("Running"):
        port = _extract_port(info)
        with _lock:
            record = _deployments.get(deployment_id)
            if record is None:
                record = {
                    "deployment_id": deployment_id,
                    "container_name": container_name,
                    "config": config.model_dump(),
                    "port": port,
                    "status": "running",
                    "warnings": [],
                    "command": [],
                }
                _deployments[deployment_id] = record
            reused = dict(record)
        reused["status"] = "running"
        reused["warnings"] = list(reused.get("warnings", [])) + [
            f"Container '{container_name}' was already running -- reused it instead of creating a new one."
        ]
        return reused

    port = _extract_port(info) or config.port
    return start_container(deployment_id, container_name, port, config.model_dump())


def reconcile(retries: int = 3, retry_delay: float = 3.0) -> None:
    """Discover existing trtllm-ui-* containers on process startup and
    rebuild the in-memory deployment table from them.

    Retried a few times (default 3, 3s apart) because right after a fresh
    Windows boot, Docker Desktop / the WSL2 Docker daemon may not have
    finished starting yet -- a single failed `docker ps` at t=0 should not
    be treated as "no deployments exist".
    """
    p = None
    for attempt in range(retries):
        p = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=trtllm-ui-", "--format", "{{json .}}"],
            capture_output=True, text=True, check=False
        )
        if p.returncode == 0:
            break
        if attempt < retries - 1:
            time.sleep(retry_delay)
    if p is None or p.returncode != 0:
        return

    for line in p.stdout.splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        name = item.get("Names", "")
        if not name.startswith("trtllm-ui-"):
            continue
        deployment_id = name.removeprefix("trtllm-ui-")

        with _lock:
            if deployment_id in _deployments:
                continue  # already tracked this process lifetime

        info = _docker_inspect(name)
        port = _extract_port(info)
        reconstructed = _reconstruct_config(info)
        running = item.get("State") == "running"
        with _lock:
            _deployments.setdefault(deployment_id, {
                "deployment_id": deployment_id,
                "container_name": name,
                "port": port,
                "config": reconstructed,
                "status": "running" if running else "exited",
                "warnings": [
                    "Rediscovered after restart -- config reconstructed from docker inspect, "
                    "not the original request. Port recovered from container's published bindings."
                ],
                "command": [] if running else ["docker", "start", name],
            })


def get_status(deployment_id: str) -> dict[str, Any]:
    with _lock:
        record = _deployments.get(deployment_id)
    if not record:
        raise KeyError(deployment_id)
    info = _docker_inspect(record["container_name"])
    if info is None:
        record["status"] = "not-found"
        return record
    state = info.get("State", {})
    status = state.get("Status", "unknown")
    record["status"] = "running" if status == "running" else status
    return record


def list_deployments() -> list[dict[str, Any]]:
    with _lock:
        ids = list(_deployments)
    return [get_status(i) for i in ids]


def stop_deployment(deployment_id: str) -> None:
    record = get_status(deployment_id)
    name = record["container_name"]
    subprocess.run(["docker", "stop", name], capture_output=True, text=True, check=False)
    subprocess.run(["docker", "rm", name], capture_output=True, text=True, check=False)
    record["status"] = "stopped"


def log_stream(deployment_id: str):
    record = get_status(deployment_id)
    name = record["container_name"]
    p = subprocess.Popen(
        ["docker", "logs", "--tail", "2000", "-f", name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        # Docker's log output is UTF-8 regardless of host OS (it often
        # contains Unicode box-drawing / block characters from progress
        # bars). Without an explicit encoding, Python's text mode falls
        # back to the platform's default locale encoding -- on Windows
        # that's typically cp1252, not UTF-8 -- and raises
        # UnicodeDecodeError the first time a non-cp1252 byte shows up
        # (e.g. the "█" characters in trtllm-serve's weight-loading
        # progress bars). errors="replace" swaps undecodable bytes for
        # "�" instead of crashing the whole log stream.
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        assert p.stdout is not None
        for line in p.stdout:
            yield line.rstrip("\n")
    finally:
        if p.poll() is None:
            p.terminate()


def preflight(settings: AppSettings, config: DeploymentConfig) -> dict[str, Any]:
    """Read-only pre-flight validation, run BEFORE `docker run` is attempted.

    Returns a structured report -- never raises for expected failure modes
    (missing model, Docker down, GPU down, port taken) so the UI can show
    all of them at once instead of stopping at the first exception. This is
    advisory: a config that comes back "feasible" can still fail at actual
    deploy time (see vram_estimator's module docstring) -- the real
    trtllm-serve/Docker/GPU stack is the final authority, not this check.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    # 1. model exists & is a usable HF checkpoint
    model = None
    try:
        model = get_model(settings, config.model_name)
        add("model", "pass", f"found at {model['host_path']}")
    except KeyError as exc:
        add("model", "fail", str(exc))

    # 2. docker daemon reachable
    docker_info = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True, check=False,
    )
    docker_ok = docker_info.returncode == 0
    add("docker", "pass" if docker_ok else "fail",
        docker_info.stdout.strip() if docker_ok else (docker_info.stderr or docker_info.stdout or "docker not reachable").strip())

    # 3. GPU reachable
    gpu = gpu_monitor.poll()
    gpu_ok = "error" not in gpu and bool(gpu.get("gpus"))
    add("gpu", "pass" if gpu_ok else "fail",
        f"{len(gpu.get('gpus', []))} GPU(s) detected" if gpu_ok else gpu.get("error", "no GPU detected"))

    # 4. capability-manifest-driven checks: kv_cache_dtype + parallelism vs detected GPU count.
    # Both are skipped (warn, not fail) if the capability probe hasn't run yet --
    # we don't know enough to judge, and that's different from judging it unsupported.
    manifest = cached_manifest(settings)
    parsed = (manifest or {}).get("parsed", {})
    if manifest is None:
        add("capability_manifest", "warn", "not probed yet -- POST /api/capabilities/probe for capability-aware checks")
    else:
        kv_options = parsed.get("kv_cache_dtype_options")
        if config.kv_cache_dtype and kv_options is not None:
            ok = config.kv_cache_dtype.lower() in [o.lower() for o in kv_options]
            add("kv_cache_dtype", "pass" if ok else "fail",
                f"'{config.kv_cache_dtype}' {'is supported' if ok else 'is not in supported options'}: {kv_options}")

        parallel_max = parsed.get("parallel_max")
        requested = max(config.tensor_parallel_size, config.pipeline_parallel_size,
                         config.context_parallel_size, config.moe_expert_parallel_size)
        if parallel_max is not None and requested > 1:
            ok = requested <= parallel_max
            add("parallelism", "pass" if ok else "fail",
                f"requested parallelism {requested} vs {parallel_max} GPU(s) detected")

    # 5. port availability (read-only check; excludes this deployment's own record, same as build_command)
    deployment_id = _slugify(config.name or config.model_name)
    with _lock:
        occupied = {did: d["port"] for did, d in _deployments.items() if did != deployment_id}
    port_in_range = 1024 <= config.port <= 65535
    port_ok = port_in_range and config.port not in occupied.values() and _port_free(config.port)
    add("port", "pass" if port_ok else "fail",
        f"port {config.port} is free" if port_ok else f"port {config.port} is out of range or already in use")

    # 6. VRAM estimate -- heuristic, see vram_estimator module docstring.
    vram = None
    if model is not None and gpu_ok:
        vram = vram_estimator.estimate(
            model, config.max_batch_size, config.max_seq_len,
            config.max_output_tokens, config.kv_cache_dtype,
        )
        try:
            total = int(gpu["gpus"][0]["memory_total"])
            used = int(gpu["gpus"][0]["memory_used"])
            free_mb = total - used
            sufficient = vram["total_estimated_mb"] <= free_mb
            add("vram", "pass" if sufficient else "warn",
                f"~{vram['total_estimated_mb']}MB estimated vs ~{free_mb}MB free (heuristic, approximate -- not exact)")
        except (KeyError, ValueError, IndexError):
            add("vram", "warn", "could not read GPU memory figures to compare against estimate")

    feasible = not any(c["status"] == "fail" for c in checks)
    return {"checks": checks, "vram_estimate": vram, "feasible": feasible}


def generated_command(deployment_id: str) -> dict[str, Any]:
    record = get_status(deployment_id)
    cmd = record.get("command", [])
    return {"argv": cmd, "command": _quote_cmd(cmd)}

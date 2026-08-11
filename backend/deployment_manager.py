from __future__ import annotations

import json
import shlex
import socket
import subprocess
import threading
import time
import uuid
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from pydantic import BaseModel, Field

from .capability_probe import cached_manifest
from .model_manager import get_model
from .settings import AppSettings


class DeploymentConfig(BaseModel):
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
    if not (1024 <= config.port <= 65535):
        raise ValueError("port must be between 1024 and 65535")
    with _lock:
        occupied = {d["port"] for d in _deployments.values()}
    if config.port in occupied or not _port_free(config.port):
        raise ValueError(f"port {config.port} is already in use")

    if config.tensor_parallel_size < 1 or config.pipeline_parallel_size < 1 or config.context_parallel_size < 1 or config.moe_expert_parallel_size < 1:
        raise ValueError("parallelism values must be >= 1")

    if (config.trust_remote_code or config.custom_module_dirs) and config.unsafe_ack != "ENABLE UNSAFE":
        raise ValueError('Unsafe flags require unsafe_ack == "ENABLE UNSAFE"')

    deployment_id = deployment_id or uuid.uuid4().hex[:12]
    container_name = f"trtllm-ui-{deployment_id}"

    model = get_model(settings, config.model_name)
    cmd = [
        "docker", "run", "-d",
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


def start_deployment(settings: AppSettings, config: DeploymentConfig) -> dict[str, Any]:
    deployment_id = uuid.uuid4().hex[:12]
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


def reconcile() -> None:
    p = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=trtllm-ui-", "--format", "{{json .}}"],
        capture_output=True, text=True, check=False
    )
    if p.returncode != 0:
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
            _deployments.setdefault(deployment_id, {
                "deployment_id": deployment_id,
                "container_name": name,
                "port": None,
                "config": {},
                "status": "running" if item.get("State") == "running" else "exited",
                "warnings": [],
                "command": [],
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
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        assert p.stdout is not None
        for line in p.stdout:
            yield line.rstrip("\n")
    finally:
        if p.poll() is None:
            p.terminate()


def generated_command(deployment_id: str) -> dict[str, Any]:
    record = get_status(deployment_id)
    cmd = record.get("command", [])
    return {"argv": cmd, "command": _quote_cmd(cmd)}

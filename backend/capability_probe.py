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


def _image_tag(image: str) -> str:
    return image.replace("/", "_").replace(":", "_").replace("@", "_")


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

        manifest = {
            "probed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image": image,
            "image_tag": _image_tag(image),
            "commands": results,
            "parsed": {
                "serve_flags": _parse_flags(
                    results[3]["stdout"] + "\n" + results[3]["stderr"]
                )
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

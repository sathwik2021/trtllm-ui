from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .capability_probe import cached_manifest
from .settings import AppSettings, is_c_drive_path, to_host_path


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _usage(path: str) -> dict[str, Any]:
    try:
        u = shutil.disk_usage(to_host_path(path))
        return {"total": u.total, "used": u.used, "free": u.free}
    except Exception as exc:
        return {"error": str(exc)}


def get_storage(settings: AppSettings) -> dict[str, Any]:
    data = {
        "c_drive": _usage("/mnt/c" if Path("/mnt/c").exists() else "C:/"),
        "d_drive": _usage("/mnt/d" if Path("/mnt/d").exists() else "D:/"),
        "model_dir": _dir_size(to_host_path(settings.model_dir)),
        "logs_dir": _dir_size(to_host_path(settings.logs_dir or "")),
        "profiles_dir": _dir_size(to_host_path(settings.profiles_dir or "")),
        "c_drive_path_warning": bool(settings.path_warnings()),
        "path_warnings": settings.path_warnings(),
    }
    manifest = cached_manifest(settings)
    if manifest:
        data["docker_system_df"] = manifest.get("commands", [None, {}])[1]
    else:
        data["docker_system_df"] = None
    return data

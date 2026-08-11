from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .settings import AppSettings, to_host_path


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip(" .")
    if not name:
        raise ValueError("invalid profile name")
    return name


def _dir(settings: AppSettings) -> Path:
    settings.materialize_dirs()
    return to_host_path(settings.profiles_dir or str(Path(settings.data_dir) / "profiles"))


def list_profiles(settings: AppSettings) -> list[str]:
    return sorted(p.stem for p in _dir(settings).glob("*.json"))


def get_profile(settings: AppSettings, name: str) -> dict[str, Any]:
    path = _dir(settings) / f"{_safe(name)}.json"
    if not path.exists():
        raise KeyError(name)
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile(settings: AppSettings, name: str, config: dict[str, Any]) -> None:
    path = _dir(settings) / f"{_safe(name)}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def delete_profile(settings: AppSettings, name: str) -> None:
    path = _dir(settings) / f"{_safe(name)}.json"
    if not path.exists():
        raise KeyError(name)
    path.unlink()

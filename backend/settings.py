from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _default_model_dir() -> str:
    return r"D:\models" if platform.system() == "Windows" else "/mnt/d/models"


def _default_data_dir() -> str:
    return r"D:\trtllm-ui" if platform.system() == "Windows" else "/mnt/d/trtllm-ui"


def to_host_path(value: str) -> Path:
    """Accept Windows D:/... and WSL /mnt/d/... paths on either host OS."""
    p = value.strip()
    if platform.system() == "Windows":
        if p.startswith("/mnt/") and len(p) >= 6:
            drive = p[5].upper()
            rest = p[6:].replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
        return Path(p)
    if len(p) >= 3 and p[1:3] == ":\\":
        drive = p[0].lower()
        return Path("/mnt") / drive / p[3:].replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return Path("/mnt") / drive / p[2:].lstrip("/").replace("\\", "/")
    return Path(p)


def is_c_drive_path(value: str) -> bool:
    p = str(to_host_path(value)).replace("\\", "/").lower()
    return p == "/mnt/c" or p.startswith("/mnt/c/")


class AppSettings(BaseModel):
    model_dir: str = Field(default_factory=_default_model_dir)
    data_dir: str = Field(default_factory=_default_data_dir)
    profiles_dir: str | None = None
    logs_dir: str | None = None
    docker_image: str = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22"
    host: str = "127.0.0.1"
    port_range_start: int = 8000

    def materialize_dirs(self) -> None:
        if self.profiles_dir is None:
            self.profiles_dir = str(Path(self.data_dir) / "profiles")
        if self.logs_dir is None:
            self.logs_dir = str(Path(self.data_dir) / "logs")
        for raw in (self.data_dir, self.profiles_dir, self.logs_dir):
            to_host_path(raw).mkdir(parents=True, exist_ok=True)

    def path_warnings(self) -> list[str]:
        warnings = []
        for label, raw in {
            "model_dir": self.model_dir,
            "data_dir": self.data_dir,
            "profiles_dir": self.profiles_dir or "",
            "logs_dir": self.logs_dir or "",
        }.items():
            if raw and is_c_drive_path(raw):
                warnings.append(f"{label} resolves under the C: drive: {raw}")
        return warnings


def settings_file(settings: AppSettings) -> Path:
    return to_host_path(settings.data_dir) / "settings.json"


def load_settings() -> AppSettings:
    raw_data = None
    data_hint = os.environ.get("TRTLLM_UI_DATA_DIR")
    if data_hint:
        raw_data = {"data_dir": data_hint}
    base = AppSettings(**(raw_data or {}))
    path = settings_file(base)
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            base = AppSettings(**stored)
        except Exception:
            pass
    base.materialize_dirs()
    return base


def save_settings(settings: AppSettings) -> None:
    settings.materialize_dirs()
    path = settings_file(settings)
    path.write_text(
        json.dumps(settings.model_dump(), indent=2),
        encoding="utf-8",
    )

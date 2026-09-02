from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .settings import AppSettings, to_host_path


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _container_model_path(name: str) -> str:
    return f"/models/{name}"


def list_models(settings: AppSettings) -> list[dict[str, Any]]:
    base = to_host_path(settings.model_dir)
    if not base.exists():
        return []
    result = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or not (child / "config.json").is_file():
            continue
        try:
            config = json.loads((child / "config.json").read_text(encoding="utf-8"))
        except Exception:
            config = {}
        tok = child / "tokenizer_config.json"
        tok_data = {}
        if tok.is_file():
            try:
                tok_data = json.loads(tok.read_text(encoding="utf-8"))
            except Exception:
                pass
        result.append({
            "name": child.name,
            "size_bytes": _size(child),
            "architectures": config.get("architectures", []),
            "model_type": config.get("model_type"),
            "torch_dtype": config.get("torch_dtype"),
            "has_chat_template": bool(tok_data.get("chat_template")),
            "host_path": str(child.resolve()),
            "container_path": _container_model_path(child.name),
        })
    return result


def get_model(settings: AppSettings, name: str) -> dict[str, Any]:
    for model in list_models(settings):
        if model["name"] == name:
            return model
    raise KeyError(f"Model not found: {name}")

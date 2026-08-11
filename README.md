# TensorRT-LLM Local Control Panel — v1

Thin localhost-only FastAPI orchestrator for the confirmed TensorRT-LLM baseline.

## Important binding distinction

There are three different network bindings:

1. FastAPI/uvicorn binds `127.0.0.1:8420` on the host.
2. Docker publishes each serving port as `127.0.0.1:<port>:<port>` on the host.
3. TensorRT-LLM receives `--host 0.0.0.0` inside the container. This is intentionally not changed to `127.0.0.1`.

No authentication is included. Do not expose this application beyond localhost.

## Install

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

WSL:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

Open `http://127.0.0.1:8420`.

## Storage

Default model directory: `D:\models` (`/mnt/d/models` under WSL).

Application data: `D:\trtllm-ui` (`/mnt/d/trtllm-ui` under WSL).

The app warns if configured application paths resolve to C:.

## v1 scope

Included: diagnostics, model discovery, command generation, Docker deployment lifecycle, logs, GPU polling, profiles API, storage API, basic chat/test UI.

Not included: HuggingFace downloads, engine building, quantization, authentication, or undocumented TensorRT-LLM workflows.

## Verification status

This source bundle can be syntax-checked locally, but the actual GPU/Docker acceptance test must be run on the target Windows/WSL2 machine. No claim is made here that a real TensorRT-LLM container was launched from this environment.

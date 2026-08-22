from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import benchmark_manager, capability_probe, config_manager, deployment_manager, gpu_monitor, model_manager, storage_manager
from .deployment_manager import DeploymentConfig
from .settings import AppSettings, load_settings, save_settings


settings = load_settings()
app = FastAPI(title="TensorRT-LLM Local Control Panel")
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "frontend"), name="static")


@app.on_event("startup")
def startup():
    settings.materialize_dirs()

    # Cold-boot handling: right after a Windows reboot, WSL2's GPU
    # passthrough and/or the Docker daemon may not be ready the instant
    # this process starts. Retry both a few times rather than failing
    # permanently on the first check and reporting "unavailable" for the
    # rest of the process's life.
    gpu_result = gpu_monitor.poll_with_retry(attempts=3, delay=3.0)
    if "error" in gpu_result:
        print(f"[startup] GPU still unavailable after retries: {gpu_result['error']}")
    else:
        print(f"[startup] GPU check OK: {len(gpu_result.get('gpus', []))} GPU(s) detected")

    deployment_manager.reconcile(retries=3, retry_delay=3.0)
    try:
        found = len(deployment_manager.list_deployments())
    except Exception as exc:
        found = f"unknown ({exc})"
    print(f"[startup] reconcile() found {found} existing deployment(s)")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")


@app.get("/api/settings")
def api_settings():
    return settings.model_dump() | {"path_warnings": settings.path_warnings()}


@app.put("/api/settings")
def api_settings_put(payload: dict):
    global settings
    try:
        updated = AppSettings(**payload)
        updated.materialize_dirs()
        save_settings(updated)
        settings = updated
        return api_settings()
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/capabilities")
def api_capabilities():
    manifest = capability_probe.cached_manifest(settings)
    if manifest is None:
        return {"status": "not_yet_probed", "message": "POST /api/capabilities/probe to run diagnostics"}
    return {"status": "ready", "manifest": manifest}


@app.post("/api/capabilities/probe")
def api_probe(background_tasks: BackgroundTasks):
    if capability_probe._probe_lock.locked():
        return {"status": "already_running"}
    background_tasks.add_task(capability_probe.run_probe, settings)
    return {"status": "started"}


@app.get("/api/models")
def api_models():
    return model_manager.list_models(settings)


@app.get("/api/models/{name}")
def api_model(name: str):
    try:
        return model_manager.get_model(settings, name)
    except KeyError:
        raise HTTPException(404, "model not found")


@app.post("/api/deployments")
def api_deploy(config: DeploymentConfig):
    try:
        return deployment_manager.start_deployment(settings, config)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/deployments/preflight")
def api_preflight(config: DeploymentConfig):
    # No try/except needed: preflight() is designed to never raise for
    # expected failure modes (missing model, docker down, etc) -- those
    # come back as checks with status "fail", not exceptions.
    return deployment_manager.preflight(settings, config)


@app.get("/api/deployments")
def api_deployments():
    return deployment_manager.list_deployments()


@app.get("/api/deployments/{deployment_id}")
def api_deployment(deployment_id: str):
    try:
        return deployment_manager.get_status(deployment_id)
    except KeyError:
        raise HTTPException(404, "deployment not found")


@app.post("/api/deployments/{deployment_id}/stop")
def api_stop(deployment_id: str):
    try:
        deployment_manager.stop_deployment(deployment_id)
        return deployment_manager.get_status(deployment_id)
    except KeyError:
        raise HTTPException(404, "deployment not found")


@app.get("/api/deployments/{deployment_id}/generated-command")
def api_command(deployment_id: str):
    try:
        return deployment_manager.generated_command(deployment_id)
    except KeyError:
        raise HTTPException(404, "deployment not found")


@app.get("/api/deployments/{deployment_id}/logs")
def api_logs(deployment_id: str):
    try:
        deployment_manager.get_status(deployment_id)
    except KeyError:
        raise HTTPException(404, "deployment not found")

    def events():
        for line in deployment_manager.log_stream(deployment_id):
            yield f"data: {json.dumps(line)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/gpu")
def api_gpu():
    return gpu_monitor.poll()


@app.get("/api/gpu/stream")
async def api_gpu_stream():
    async def events():
        while True:
            yield f"data: {json.dumps(gpu_monitor.poll())}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/benchmarks")
def api_start_benchmark(req: benchmark_manager.BenchmarkRequest):
    host = req.host
    port = req.port
    served_model_name = req.served_model_name

    if req.deployment_id:
        try:
            record = deployment_manager.get_status(req.deployment_id)
        except KeyError:
            raise HTTPException(404, f"deployment not found: {req.deployment_id}")
        host = host or settings.host
        port = port or record["port"]
        served_model_name = served_model_name or (record.get("config") or {}).get("served_model_name")

    if not (host and port and served_model_name):
        raise HTTPException(
            400,
            "need either deployment_id, or explicit host+port+served_model_name",
        )

    try:
        return benchmark_manager.start_benchmark_job(
            settings, host=host, port=port, served_model_name=served_model_name,
            request_count=req.request_count, concurrency=req.concurrency,
            max_tokens=req.max_tokens, prompt=req.prompt, request_timeout=req.request_timeout,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/benchmarks/status")
def api_benchmark_status():
    job = benchmark_manager.current_job()
    return job or {"status": "idle"}


@app.get("/api/benchmarks")
def api_list_benchmarks():
    return benchmark_manager.list_results(settings)


@app.get("/api/benchmarks/{benchmark_id}")
def api_get_benchmark(benchmark_id: str):
    try:
        return benchmark_manager.get_result(settings, benchmark_id)
    except KeyError:
        raise HTTPException(404, "benchmark not found")


@app.delete("/api/benchmarks/{benchmark_id}")
def api_delete_benchmark(benchmark_id: str):
    benchmark_manager.delete_result(settings, benchmark_id)
    return {"deleted": benchmark_id}


@app.get("/api/storage")
def api_storage():
    return storage_manager.get_storage(settings)


@app.get("/api/profiles")
def api_profiles():
    return config_manager.list_profiles(settings)


@app.get("/api/profiles/{name}")
def api_profile(name: str):
    try:
        return config_manager.get_profile(settings, name)
    except KeyError:
        raise HTTPException(404, "profile not found")


@app.post("/api/profiles")
def api_save_profile(payload: dict):
    try:
        config_manager.save_profile(settings, payload["name"], payload["config"])
        return {"status": "saved"}
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/profiles/{name}")
def api_delete_profile(name: str):
    try:
        config_manager.delete_profile(settings, name)
        return {"status": "deleted"}
    except KeyError:
        raise HTTPException(404, "profile not found")

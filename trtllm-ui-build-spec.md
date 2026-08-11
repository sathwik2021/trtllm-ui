# TensorRT-LLM Local Control Panel — Build Specification (v1: Thin Orchestrator)

Paste this entire document into a new chat with an AI coding assistant to build the application.

## Instructions to the coding AI — read before anything else

You are implementing this project, not redesigning it. Treat the following specification as authoritative v1 scope.

- Do not add TensorRT-LLM features marked v2.
- Do not assume undocumented TensorRT-LLM behavior.
- Do not replace confirmed working commands with commands from older TensorRT-LLM documentation.
- Do not silently change storage locations.
- Do not introduce npm, React, a database, the Docker SDK, or any dependency beyond `requirements.txt` unless there is a concrete implementation blocker — state the blocker before adding anything.
- Before writing code, inspect this specification for internal contradictions and name them. In particular: distinguish FastAPI's host binding, Docker's port-publish binding, and TensorRT-LLM's internal `--host` flag — these are three separate things and must not be conflated.
- Implement the smallest complete v1 that passes the acceptance test.
- Build and verify incrementally (see "Incremental build order" near the end) — do not write all files in one pass without running anything in between.
- After each step, report what you implemented, what you tested, and what remains unverified. Do not claim something "works" because code was written for it — only claim it works after actually running it and observing the result.
- If something fails during implementation, debug the actual failure. Do not paper over it by adding speculative functionality.

## Scope honesty clause — read first

This spec builds ONLY what is confirmed to work in the current environment. It deliberately excludes:
- Engine building / `trtllm-build` workflows (binary confirmed absent from image; unconfirmed whether an alternate Python API path exists)
- Quantization (blocked anyway — `transformers 5.5.4` confirmed incompatible with `nvidia-modelopt`)
- Assumptions about GPU precision support (FP8/NVFP4 require specific GPU architectures — GPU model was never confirmed in this project's diagnostics, so the app must detect this at runtime, never hardcode it)
- Multi-GPU parallelism as a "just works" feature (TP/PP/CP/EP flags are exposed, but the app must query actual GPU count at runtime and block invalid configs, never assume >1 GPU exists)

This is a deliberate scope cut, not a limitation of ambition. Building against unconfirmed capabilities produces code that fails on first run. This spec only uses facts already verified by a real successful deployment (see "Confirmed baseline" below). Extending to engine-build/quantization is a v2 task, after running the diagnostic script included here and confirming those paths exist.

## Confirmed baseline (do not deviate from these facts)

- Host: Windows + WSL2 + Docker Desktop 4.85.0 (Engine 29.6.2)
- Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`
- Available executables in image: `trtllm-serve`, `trtllm-bench`, `trtllm-eval`, `trtllm-llmapi-launch` (NOT `trtllm-build`, NOT `convert_checkpoint.py`)
- CLI flags use underscores, not hyphens (e.g. `--served_model_name`, confirmed by testing; hyphenated form errors out)
- Confirmed working launch command:
```
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v /mnt/d/models/Qwen2.5-1.5B-Instruct:/models/Qwen2.5-1.5B-Instruct:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22 \
  trtllm-serve serve \
  /models/Qwen2.5-1.5B-Instruct \
  --backend pytorch \
  --host 0.0.0.0 \
  --port 8000 \
  --served_model_name Qwen2.5-1.5B-Instruct
```
- Confirmed working test model: `Qwen2.5-1.5B-Instruct` at `D:\models\Qwen2.5-1.5B-Instruct` (`/mnt/d/models/Qwen2.5-1.5B-Instruct` in WSL)
- Confirmed working API test: `POST /v1/chat/completions`, OpenAI-compatible response shape
- **Note on the command above:** `-p 8000:8000` in the tested command publishes the container port on all host interfaces. This spec's generated commands must instead use `-p 127.0.0.1:8000:8000` (see Hard Constraint 3 below) — a different binding than what was manually tested, but equivalent in behavior for a localhost-only workflow. `--host 0.0.0.0` stays unchanged inside the container; that's the container's internal listen address, unrelated to the host publish binding.
- Confirmed CLI flags that exist on `trtllm-serve serve` (full help text/choices not yet captured — app must not hardcode value ranges for these, only flag names):
  `--tokenizer --custom_tokenizer --post_processor_hook --host --port --backend --custom_module_dirs --log_level --max_beam_width --max_batch_size --max_num_tokens --max_seq_len --tensor_parallel_size --pipeline_parallel_size --context_parallel_size --moe_expert_parallel_size --gpus_per_node --free_gpu_memory_fraction --kv_cache_dtype --num_postprocess_workers --num_input_processor_workers --num_media_load_workers --trust_remote_code --hf_revision --config --reasoning_parser --tool_parser --metadata_server_config_file --server_role --enable_chunked_prefill --enable_attention_dp --media_io_kwargs --video_pruning_rate --chat_template --allow_request_chat_template --middleware --grpc --served_model_name --enable_visual_gen --visual_gen_args --agent_percentage --agent_types`

## Hard constraints

1. All application data (profiles, logs, cache) lives under `D:\trtllm-ui\` (WSL: `/mnt/d/trtllm-ui/`). Never write to `C:` except the Python venv/app code itself.
2. Model directory defaults to `D:\models`, user-configurable, never hardcoded elsewhere.
3. Two separate bindings must both be localhost-only, and must not be confused with each other:
   - FastAPI/uvicorn itself binds `127.0.0.1:8420` (the control panel UI).
   - Every Docker container launched by the app publishes ports as `-p 127.0.0.1:<port>:<port>`, never `-p <port>:<port>` (which would expose it on all network interfaces). The container's internal `--host 0.0.0.0` (TensorRT-LLM's own listen address inside its network namespace) is correct and unrelated — do not change it to `127.0.0.1`, that would break the container's own binding.
   No auth system in v1 — this is explicitly a localhost-only tool, and the README must say so.
4. Every Docker invocation is built as a Python list (`subprocess.run([...])`), never a shell string. No `shell=True` anywhere in the codebase.
5. `--trust_remote_code` and `--custom_module_dirs` are never set by default. Enabling either requires the user to type the literal string `ENABLE UNSAFE` into a confirmation field in the UI before the flag is included in the generated command. A persistent red banner must be shown on any deployment using either flag: "This deployment can execute arbitrary code. Only use with models/modules you trust."
6. The exact command about to run is always shown to the user before every launch (copyable text block), regardless of whether Normal or Expert mode was used to build it.

## Tech stack (deliberately minimal — every added dependency is a potential one-shot-build failure point)

- Backend: Python 3.11+, FastAPI, `uvicorn[standard]`, no ORM, no database — JSON files on disk.
- Frontend: single static HTML file + vanilla JS + minimal CSS, served by FastAPI's `StaticFiles`. No React, no npm, no build step. This is a deliberate simplification for v1 — a build step is a common source of "doesn't run in one go" failures (dependency resolution, lockfile drift, Node version mismatches). Revisit only if the vanilla JS becomes unmaintainable.
- Docker interaction: Python `subprocess` module directly, not `docker-py` SDK — fewer version-compatibility surfaces to break.
- Live logs / GPU stats: Server-Sent Events (`text/event-stream`), not WebSocket — simpler to implement correctly in FastAPI without extra deps.

## Directory structure to create

```
trtllm-ui/
├── backend/
│   ├── main.py                  # FastAPI app, mounts routes + static files
│   ├── capability_probe.py      # runs diagnostics, writes capability manifest JSON
│   ├── model_manager.py         # scans model dir, reads config.json metadata
│   ├── deployment_manager.py    # builds docker command, launches/stops/tracks process
│   ├── gpu_monitor.py           # parses `nvidia-smi --query-gpu=...` on an interval
│   ├── config_manager.py        # profile save/load/list/delete as JSON files
│   ├── storage_manager.py       # disk usage checks via shutil.disk_usage
│   └── settings.py              # paths config (model dir, data dir, defaults), loaded from env or a settings.json
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
├── README.md
└── run.py                       # entrypoint: `python run.py`, starts uvicorn on 127.0.0.1:8420
```

## requirements.txt (pin these exact minimums)

```
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.7
```

Nothing else. Do not add packages beyond this without a stated reason.

## Backend module responsibilities

### `settings.py`
- Defines `AppSettings` (pydantic model): `model_dir`, `data_dir`, `profiles_dir`, `logs_dir`, `docker_image`, `host`, `port_range_start`.
- Defaults: `model_dir=D:/models` (as WSL path `/mnt/d/models`), `data_dir=D:/trtllm-ui` (`/mnt/d/trtllm-ui`), `docker_image=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`.
- On startup, creates `data_dir`, `profiles_dir`, `logs_dir` if missing. If any configured path resolves under `/mnt/c/` or `C:\`, log a visible warning (do not silently allow).
- Settings are editable via `GET/PUT /api/settings`, persisted to `data_dir/settings.json`.

### `capability_probe.py`
- Function `run_probe() -> dict` that executes, in order, and captures stdout/stderr/exit code for each (never raise on non-zero exit — record failure and continue):
  1. `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv`
  2. `docker system df --format json` (fallback to plain `docker system df` if `--format json` unsupported)
  3. `docker info --format '{{.DockerRootDir}}'`
  4. `docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 <image> trtllm-serve serve --help`
  5. `docker run --rm --gpus all <image> trtllm-serve --help`
- Parses (4)'s output into a flag list: flag name, whether it takes a value, help text line if extractable. Store raw text alongside parsed result — never discard the raw capture.
- Writes result to `data_dir/capability_manifest_<image_tag>.json`, including a `probed_at` timestamp.
- Exposed via `GET /api/capabilities` (returns cached manifest if present, otherwise instructs caller to `POST /api/capabilities/probe` to run it — probing takes tens of seconds, must not block other requests, run in a background task).
- This module is the single source of truth for what flags/GPU facts exist. No other module hardcodes GPU capability assumptions.

### `model_manager.py`
- `list_models() -> list[dict]`: scans `settings.model_dir` one level deep for subdirectories containing a `config.json`. For each, reads `config.json` for `architectures`, `model_type`, `torch_dtype` if present; reads `tokenizer_config.json` if present for `chat_template` presence; computes total directory size via `os.walk` + `os.path.getsize`.
- `get_model(name) -> dict`: full detail for one model, including full absolute host path and the path as it will appear inside the container.
- No download/HuggingFace-fetch functionality in v1 — out of scope for this pass, note as v2 (do not stub a broken button for it in the UI; either omit the page entirely or clearly mark it "not yet implemented").

### `deployment_manager.py`

Docker is the source of truth for whether a deployment exists and is running — not the Python process that launched it. This matters because the FastAPI process can restart (dev reload, crash) while containers keep running; status must be recoverable from `docker inspect`/`docker ps`, not lost with the Python process's memory.

- Lifecycle: `deployment_id` → `container_name = f"trtllm-ui-{deployment_id}"` → `docker run --name {container_name} ...` (launched via `subprocess.run(cmd, check=False)` in `--detach` mode, i.e. add `-d` to the docker args instead of using `docker run` in the foreground) → status/logs/stop all operate via `docker inspect {container_name}`, `docker logs {container_name}`, `docker stop {container_name}`.
- In-memory dict still exists (`{deployment_id: {config: dict, container_name: str, port: int}}`) but only for config/metadata the app needs — **never as the source of truth for running/not-running**. On FastAPI startup, reconcile this dict by running `docker ps -a --filter name=trtllm-ui- --format json` and rebuilding known deployments from actual running containers.
- `build_command(config: DeploymentConfig) -> list[str]`: returns the full `docker run` argv list, including `-d` (detached) and `--name {container_name}`. Validates:
  - `config.model_name` exists in `model_manager.list_models()`
  - `config.port` not already in use by another tracked deployment or by a `socket.bind` test
  - port publish is always built as `-p 127.0.0.1:{port}:{port}` — never bind to all interfaces
  - if `config.trust_remote_code` or `config.custom_module_dirs` set, requires `config.unsafe_ack == "ENABLE UNSAFE"` or raises `ValueError`
  - every flag key in `config.extra_flags` (Expert mode raw additions) is checked against the capability manifest; unknown flags are allowed but returned in a `warnings` list, never silently dropped or silently blocked
- `start_deployment(config) -> deployment_id`: runs `build_command`, launches via `subprocess.run(cmd, capture_output=True, text=True)` (detached container, so this call returns quickly once the container starts, not once TRT-LLM finishes loading). Sets status `starting`. A separate watcher polls `GET /v1/models` against the deployment's port every 2s up to 90s; on success set status `ready`; on timeout set status `error` with reason "startup timeout — check logs"; also treat `docker inspect` reporting the container as `Exited` as an immediate error, don't wait out the full timeout in that case.
- `get_status(deployment_id)`: queries `docker inspect {container_name}` fresh each call (state: running/exited/not-found) rather than trusting cached in-memory status alone — reconcile and update the in-memory status if they disagree.
- `stop_deployment(deployment_id)`: `docker stop {container_name}`, then `docker rm {container_name}` (container was run with `-d`, not `--rm`, specifically so `docker logs` remains available for post-mortem after a crash — clean up explicitly on stop instead).
- `get_logs(deployment_id)`: `docker logs --tail 2000 -f {container_name}`, streamed line-by-line into the SSE response — read directly from Docker, not from an app-maintained buffer, so logs survive an app restart.
- `GET /api/deployments`, `POST /api/deployments`, `POST /api/deployments/{id}/stop`, `GET /api/deployments/{id}/logs` (SSE stream), `GET /api/deployments/{id}/generated-command` (returns the exact argv list joined for display, plus a copy-pasteable single-line string).

### `gpu_monitor.py`
- `poll() -> dict`: runs `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits`, parses CSV, returns structured dict per GPU index. On failure (e.g. command not found, GPU busy), returns `{"error": "<message>"}`, never raises to caller.
- `GET /api/gpu/stream` — SSE endpoint emitting `poll()` result every 2 seconds.

### `config_manager.py`
- Profiles are JSON files in `data_dir/profiles/<name>.json`, containing the full `DeploymentConfig`.
- `GET /api/profiles`, `GET /api/profiles/{name}`, `POST /api/profiles`, `DELETE /api/profiles/{name}`.
- No versioning/migration logic in v1 — if a profile references a flag no longer in the capability manifest, surface a warning in the UI when loading it, don't block loading.

### `storage_manager.py`
- `GET /api/storage`: returns `shutil.disk_usage` for `C:\` and `D:\` (or their WSL mount points), plus sizes of `model_dir`, `data_dir/logs`, `data_dir/profiles`, and Docker's reported usage from the cached capability manifest (`docker system df` output). Flags in the response if any configured path resolves to the C: drive.

## Frontend pages (single HTML file, JS-driven view switching — no router library)

1. **Dashboard** — capability manifest summary (GPU name/VRAM, TRT-LLM version/image tag, Docker root dir), storage summary, list of active deployments with status.
2. **Models** — table from `GET /api/models`: name, size, architecture, path. Row action: "Deploy" (pre-fills deployment form).
3. **Deploy** — form for `DeploymentConfig`: model select, backend (dropdown populated from capability manifest if parseable, else free text), host/port, served_model_name, max_batch_size, max_seq_len, tensor_parallel_size (with inline warning if > detected GPU count), kv_cache_dtype, free_gpu_memory_fraction. Collapsible "Advanced" section for the remaining confirmed flags. Collapsible "Expert" section: raw key-value pairs appended verbatim, plus the trust_remote_code/custom_module_dirs fields gated by the `ENABLE UNSAFE` text input. Always shows a live-updating "Generated command" preview box above the Deploy button.
4. **Deployment detail** — status, live log stream (SSE), Stop/Restart buttons, link to Chat page pre-configured with this deployment's port.
5. **Chat / Test** — model dropdown (running deployments only), message box, temperature/top_p/max_tokens fields, sends to the deployment's `/v1/chat/completions`, displays response plus prompt/completion token counts and latency if present in the API response.
6. **Storage** — disk usage bars for C:/D:, breakdown of app data dirs, warning banner if anything resolves to C:.
7. **Profiles** — list/save/load/delete.
8. **Diagnostics** — button to re-run `capability_probe.run_probe()`, displays raw output of each probed command in collapsible panels (this is where the user reads the full `trtllm-serve serve --help` text, `nvidia-smi` output, etc., unfiltered).

No "Normal / Advanced / Expert mode" as three separate top-level toggles — collapse them into progressive disclosure within the single Deploy form (Basic fields visible, Advanced/Expert collapsed by default), per the reasoning that a hard 3-tier mode switch implied a novice-user persona that doesn't clearly exist for this tool.

## Incremental build order — follow this, do not skip ahead

Each step must be run and observed working before starting the next. This isolates failures to a single layer.

**Step 1 — skeleton.** Create the directory tree, `requirements.txt`, `settings.py`, minimal `main.py` serving the static frontend, `/api/settings` (GET/PUT), and a Dashboard page that just shows current settings values. Run `python run.py`, confirm the page loads at `127.0.0.1:8420` with no errors.

**Step 2 — diagnostics.** Implement `capability_probe.py` and the Diagnostics page. Run it for real against the actual machine. Confirm GPU info, Docker info, and the full `trtllm-serve serve --help` text appear correctly in the UI. This is the step that turns unconfirmed facts (GPU model, exact flag list) into confirmed ones — do not proceed until this has actually run once.

**Step 3 — model discovery.** Implement `model_manager.py` and the Models page. Confirm it lists `Qwen2.5-1.5B-Instruct` from `D:\models` with correct size/metadata.

**Step 4 — command generation only, no launch.** Implement the Deploy form and `build_command()`, but do not wire up `start_deployment` yet. Display the generated command and manually diff it against the confirmed working baseline command (accounting for the `-p 127.0.0.1:8000:8000` change). They should match on every other flag.

**Step 5 — actual launch.** Wire up `start_deployment`/`stop_deployment` with the Docker-source-of-truth lifecycle described above. Launch the real container, confirm `starting` → `ready` transition, hit `/v1/models` and `/v1/chat/completions` for real, confirm `docker ps` and `docker logs` match what the UI shows. Stop it, confirm `docker ps -a` shows a clean exit, no orphaned container.

**Step 6 — remaining features.** Only after Step 5 works end-to-end: profiles, storage page, GPU monitoring stream, Expert-mode unsafe-flag gating, multi-deployment support.

## Acceptance test (must pass — this is the definition of "done" for v1)

1. Start app: `python run.py`, open `http://127.0.0.1:8420`.
2. Dashboard loads without error even if `capability_probe` has never been run (shows "not yet probed" state, not a crash).
3. Click "Run Diagnostics" — completes within ~60s, dashboard populates with real GPU/Docker/image info.
4. Models page shows `Qwen2.5-1.5B-Instruct` scanned from `D:\models`.
5. Deploy page, select that model, defaults produce a command matching the confirmed working baseline command above (backend=pytorch, host=0.0.0.0, port=8000, served_model_name=Qwen2.5-1.5B-Instruct). Click Deploy.
6. Deployment status goes `starting` → `ready` within 90s (matches observed real startup time).
7. Logs page shows the real `Started server process` / `Application startup complete` lines.
8. Chat page: send "Hello! Explain what you are in one sentence." — receive a real completion.
9. Stop deployment — status goes to `stopped`, container removed (`docker ps -a` shows no lingering `trtllm-ui-*` container in a non-exited unclean state).
10. Save this config as a profile, delete the deployment, reload profile, redeploy — same result.

If any of these fail, that is the bug list — not a reason to add unconfirmed features to compensate.

## After implementation — required report format

Do not summarize this as "done." Report explicitly, per feature:
- **Implemented** — code exists
- **Tested** — you ran it and observed output
- **Confirmed working** — matches expected behavior in the acceptance test
- **Needs testing** — code exists but wasn't actually run/verified

Also provide: complete final file tree, full contents of every created file, exact install commands, exact startup command, the acceptance-test walkthrough with actual observed results (not assumed), any assumptions made where the spec was ambiguous, and anything left unverified.

## Explicit v2 backlog (do not build now)

- HuggingFace model download
- Engine building workflow (pending confirmation of a build path — probe for it first: `python3 -c "import tensorrt_llm; print([x for x in dir(tensorrt_llm) if 'build' in x.lower()])"` inside the container, and `pip show -f tensorrt_llm | grep -i bin`)
- Quantization (blocked by transformers/modelopt conflict — resolve that first)
- Multi-model / multi-port orchestration beyond what `deployment_manager`'s dict already supports (it does support multiple concurrently, this just hasn't been UI-tested with 2+ real deployments)
- Authentication (only add if the server will ever bind beyond 127.0.0.1)

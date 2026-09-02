# TensorRT-LLM Local Control Panel

A localhost-only web app for running and operating TensorRT-LLM on a
single consumer GPU: deploy a model, benchmark it under load, tune its
config against real measurements, and have it survive a reboot without
any manual steps. Built and validated end-to-end on an RTX 3050 Laptop
(6GB) serving Qwen2.5-1.5B-Instruct via `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`.

FastAPI backend, vanilla JS frontend, no build step, no database (JSON
files on disk). 87 tests, all passing.

## What it does

- **Deploy** an HF-format model checkpoint straight into a Docker
  container running `trtllm-serve`, with a proper lifecycle: create a
  new container if none exists, `docker start` an existing stopped one,
  or reuse one that's already running — never blindly re-runs `docker
  run` into a name collision.
- **Survive a reboot with zero manual steps**: deployments use
  `--restart unless-stopped`, so Docker brings the container back on its
  own; a Windows Scheduled Task brings the app itself back at logon;
  and the app self-heals its own view of what's running (via
  `docker inspect`, not memory) if it ever misses a container at
  startup. Verified against a real cold boot, not just unit tests —
  see `scripts/PHASE1_STATUS.md`.
- **Probe real hardware capability before offering it in the UI**: runs
  `nvidia-smi` and the image's own `trtllm-serve --help` output, then
  derives what's actually usable — FP8/NVFP4 are hardware-excluded
  below Ada Lovelace/Blackwell and are omitted from the UI entirely,
  not just shown disabled. Every capability is labeled with how
  confident the app actually is in it (`confirmed` vs
  `hardware_only, needs runtime testing`), not asserted as fact.
- **Pre-flight check a deployment before touching Docker**: model
  exists, Docker/GPU reachable, requested precision/KV-cache dtype and
  parallelism are within what the probed capability manifest supports,
  port is free, and a VRAM estimate (real KV-cache math from the
  model's own `config.json` — heads, layers, hidden size — not a flat
  guess) against currently-free GPU memory.
- **Benchmark a running deployment**: fixed request count and target
  tokens, concurrency sweep, wall-clock throughput, p50/p95 latency,
  and GPU utilization/VRAM sampled for the duration of the run — same
  methodology end-to-end so results are comparable across configs.
- **Automated, statistically-aware config tuning**
  (`backend/build_sweep.py`): runs a config-variant sweep unattended —
  redeploys, waits for readiness, benchmarks, repeats each variant
  multiple times — and ranks results by median with a Mann-Whitney U
  test against baseline, because this project measured 50%+ run-to-run
  throughput noise with *zero* config change and refused to rank
  variants off a single sample after that.
- **A from-evidence decision on whether engine-build is worth
  pursuing** (`PHASE5_STATUS.md`): checked GPU hardware exclusions,
  confirmed the deployment is compute-saturated (97-98% GPU utilization
  from concurrency 250 onward) via real benchmark data, ran the config
  sweep above to confirm no lever closes a meaningful gap, and checked
  current upstream release notes directly rather than trusting
  documentation that might lag behind. Conclusion: don't pursue it, and
  the reasoning is falsifiable, not asserted.

## Important binding distinction

Three different network bindings are in play, and they're intentionally
different from each other:

1. FastAPI/uvicorn binds `127.0.0.1:8420` on the host — the control
   panel itself is never reachable off this machine.
2. Docker publishes each deployment's serving port as
   `<publish_host>:<port>:<port>`, defaulting to `127.0.0.1` (loopback
   only). Setting `publish_host` to `0.0.0.0` opts a specific
   deployment into being reachable from other devices on the LAN — the
   app surfaces an explicit warning when you do this, since there's no
   authentication anywhere in this stack.
3. TensorRT-LLM receives `--host 0.0.0.0` *inside* the container. This
   is correct and intentional — it's what lets Docker's own port
   publishing (point 2) work at all — and is not the same setting as
   point 2.

No authentication is included anywhere. Don't set `publish_host` to
anything but loopback unless you understand and accept that trade-off.

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

To have the app start itself automatically at Windows logon (so a
reboot needs zero manual steps), see `scripts/install-autostart.ps1`
and `scripts/REBOOT_TEST_CHECKLIST.md`.

## Storage

Default model directory: `D:\models` (`/mnt/d/models` under WSL).

Application data (capability manifests, benchmark results, deployment
profiles): `D:\trtllm-ui` (`/mnt/d/trtllm-ui` under WSL).

The app warns if configured application paths resolve to `C:`.

## API surface

26 endpoints across settings, capability probing, model discovery,
deployment lifecycle (deploy / preflight / status / stop / logs /
generated-command), live GPU polling, benchmarking (start / status /
list / get / delete), and named config profiles. See `backend/main.py`
for the full route table.

## Scope

**In:** capability-driven deployment (precision/KV-cache/parallelism
gated on actually-detected hardware), VRAM pre-flight estimation,
Docker lifecycle with reboot persistence, live GPU monitoring, log
streaming, load benchmarking with GPU sampling, an automated
statistical config-tuning sweep, named deployment profiles.

**Out, on purpose:** HuggingFace downloads (bring your own local
checkpoint), TensorRT engine building (investigated and explicitly
rejected for this hardware/model/release combination — see
`PHASE5_STATUS.md`, not an oversight), quantization tooling beyond what
`trtllm-serve` exposes natively, authentication (localhost-only by
design).

## Status / process docs

Each project phase closed out with a written status doc rather than
just a commit message, including what was actually tested vs. merely
implemented, and — for the reboot-persistence and engine-build
questions specifically — real evidence with file paths, not narrated
claims:

- `scripts/PHASE1_STATUS.md` — Docker/app reboot persistence, closed
  out against a real cold boot.
- `scripts/REBOOT_TEST_CHECKLIST.md` — the manual verification
  procedure used for the above.
- `PHASE5_STATUS.md` — the engine-build investigation and decision.

## Verification status

Deployment, benchmarking, and reboot-persistence have all been run
against real hardware (RTX 3050 Laptop, Windows 11 + Docker Desktop +
WSL2) — this isn't just syntax-checked. The 87 automated tests use
mocked `docker`/`nvidia-smi` calls (no GPU/Docker available in CI-style
test runs); real-environment confirmation is tracked separately in the
status docs above, not assumed from passing tests alone.
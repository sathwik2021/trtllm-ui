# Phase 1 status report

Classification key (as instructed):
- **IMPLEMENTED** — code written, not yet run at all.
- **TESTED** — ran once, in this session, against mocks (no real
  Docker/GPU/Windows available in this sandbox).
- **CONFIRMED WORKING** — verified against the real target environment
  (real Docker, real GPU, real Windows reboot). Nothing in this delivery
  is CONFIRMED WORKING yet — that requires you to run
  `scripts/REBOOT_TEST_CHECKLIST.md` on the actual machine.
- **NEEDS TESTING** — written but not exercised at all in this session
  (e.g. PowerShell, which isn't available in this sandbox).

## 1. Reconciliation (deployment_manager.py: reconcile, _extract_port, _reconstruct_config)

- Port recovery from `docker inspect` (`NetworkSettings.Ports`, with a
  `Config.Cmd` flag-parse fallback): **TESTED** — 3 unit tests, including
  the exact bug (`port: None` after restart) reproduced and fixed.
- Config reconstruction: **TESTED**, but scoped intentionally — only
  `backend`, `served_model_name`, and `image` are recovered from
  `Config.Cmd`; the record is tagged `"reconstructed": True` so the UI/API
  consumer can tell this apart from a config that came from an actual
  deploy request. Full DeploymentConfig fields (parallelism sizes, etc.)
  are NOT reconstructed — not attempted, and not claimed.
- Cold-boot retry on `docker ps` (3 attempts, 3s apart): **TESTED**
  against simulated "daemon not up yet" failures.

## 2. Create / start / reuse lifecycle (deployment_manager.py)

- Deployment identity is now derived deterministically from
  `config.name` (new optional field) or `config.model_name`, slugified —
  not a fresh random UUID per call. This is what makes "does a container
  by this name already exist" a meaningful question.
- `create_container()`: **TESTED** — only called when no container by
  that name exists.
- `start_container()`: **TESTED** — calls `docker start`, not
  `docker run`, when the container exists but isn't running.
- Reuse path in `start_deployment()`: **TESTED** — this is the actual
  gap called out in the spec (previously, calling deploy twice for the
  same effective deployment always tried `docker run` with a new random
  name, so "reuse" was structurally impossible). Confirmed via test that
  `subprocess.run` is never called at all when the container is already
  running.
- Port-collision check in `build_command()` now excludes the current
  deployment's own (possibly stale) record, so re-deploying onto the same
  name/port doesn't falsely collide with itself: **TESTED**.

## 3. Docker persistence flags

- `--restart unless-stopped` added to `docker run`: **TESTED** (asserted
  present in generated argv).
- `--rm` absent: **CONFIRMED already correct** in the original code,
  unchanged, reconfirmed by test.

## 4. Cold-boot handling

- `gpu_monitor.poll_with_retry(attempts=3, delay=3.0)`: **TESTED** in
  isolation, and **TESTED end-to-end** by directly invoking the real
  `main.startup()` function with `nvidia-smi` mocked to fail twice then
  succeed — confirmed it recovers instead of reporting permanent failure.
  The always-live `/api/gpu` endpoint intentionally still uses the
  original single-shot `poll()` — retrying every live poll would make a
  genuinely-down GPU look laggy instead of clearly erroring.
- `reconcile(retries=3, retry_delay=3.0)`: same treatment, same
  end-to-end startup test.

## 5. UI auto-start (Windows)

- **Mechanism chosen: Scheduled Task, trigger = at logon** (not a
  Windows service). Reasoning and the trade-off (won't start before an
  interactive logon) are written directly into
  `scripts/install-autostart.ps1`'s header comment, per the instruction
  to state this decision rather than default to one silently.
- `scripts/install-autostart.ps1`, `scripts/uninstall-autostart.ps1`:
  **NEEDS TESTING** — PowerShell isn't available in this sandbox
  (no `pwsh`/`powershell`, and Microsoft's package repos aren't reachable
  from this environment's allowed domain list), so these were written and
  manually reviewed but never executed. Run them on the real machine
  before trusting them.

## 6. Real reboot test

**Not performed — cannot be performed here.** No Windows, Docker, or GPU
in this sandbox. `scripts/REBOOT_TEST_CHECKLIST.md` is the concrete,
checkbox-by-checkbox procedure for you to run on the actual machine. Per
the instruction: if any item in that checklist fails, stop and report
which one rather than re-attempting fixes and re-claiming success without
another verified reboot.

## Files changed

- `backend/deployment_manager.py` — reconcile/port-recovery,
  create/start/reuse lifecycle, `--restart unless-stopped`, new optional
  `name` field on `DeploymentConfig`.
- `backend/gpu_monitor.py` — added `poll_with_retry()`.
- `backend/main.py` — startup event now uses both retry paths and logs
  the outcome instead of failing silently.
- `backend/test_deployment_manager_phase1.py` — new, 12 tests, all
  passing against mocks.
- `scripts/install-autostart.ps1`, `scripts/uninstall-autostart.ps1`,
  `scripts/REBOOT_TEST_CHECKLIST.md`, `scripts/PHASE1_STATUS.md` — new.

## Explicitly out of scope for Phase 1 (unchanged)

Everything from the analysis doc's sections 6–10: engine-build detection,
benchmark manager, visual builder, backend plugin abstraction. Not
touched, per the phased-execution instruction.

## What Phase 2 should not start until

All six items in `REBOOT_TEST_CHECKLIST.md` are checked off on the real
machine. If you hit a failure, come back with the specific checklist item
and the actual observed output before we debug further.

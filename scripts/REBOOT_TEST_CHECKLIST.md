# Phase 1 reboot verification checklist

This cannot be run in a sandbox — it requires the real Windows machine,
Docker Desktop, WSL2, and the RTX 3050. Run through this yourself after
installing the changes. Do **not** consider Phase 1 done until every box
below is checked on an actual reboot, not just "the code looks right."

## 0. One-time setup (before the reboot test)

- [ ] Pull the updated `backend/deployment_manager.py`, `backend/main.py`,
      `backend/gpu_monitor.py` into your local `trtllm-ui` checkout.
- [ ] `pip install -r requirements.txt` in your venv (no new deps were
      added, but re-run to be safe).
- [ ] Confirm `python run.py` still starts cleanly and the UI loads at
      `http://127.0.0.1:8420`.
- [ ] Deploy one model through the UI as normal, wait for status = `ready`.
      Note the **container name** (`trtllm-ui-<slug>`) and **port**.
- [ ] From an elevated PowerShell prompt, in the repo root, run:
      ```
      .\scripts\install-autostart.ps1
      ```
      Confirm it prints "Installed." with no errors.
- [ ] Test the task without rebooting first:
      ```
      Start-ScheduledTask -TaskName 'trtllm-ui-autostart'
      ```
      Wait ~10s, then confirm `http://127.0.0.1:8420` responds. If it
      doesn't, fix this before doing a real reboot — a failing
      Scheduled Task is easier to debug live than after a reboot.

## 1. The actual reboot test

- [ ] **Reboot Windows** (not sleep/hibernate — a full restart).
- [ ] Log back in normally (the task triggers at logon, so you do need
      to log in — see the trade-off documented in
      `install-autostart.ps1`'s header comment).
- [ ] Wait ~60 seconds after logon (30s task delay + startup + Docker
      warm-up).
- [ ] **Docker Desktop came up on its own** — check its own "start on
      login" setting is enabled in Docker Desktop settings; this repo's
      scripts don't control that.
- [ ] **The deployment container is running again automatically**, with
      no manual `docker start`:
      ```
      docker ps --filter "name=trtllm-ui-"
      ```
      Confirm `STATUS` shows `Up ...` for your container, and that it
      restarted on its own (check `docker inspect <name>` →
      `.State.StartedAt` is close to boot time, not something you
      triggered by hand).
- [ ] **FastAPI started automatically** — confirm the scheduled task ran:
      ```
      Get-ScheduledTask -TaskName 'trtllm-ui-autostart' | Get-ScheduledTaskInfo
      ```
      `LastTaskResult` should be `0`, `LastRunTime` should be close to
      your logon time.
- [ ] **The UI is reachable**: open `http://127.0.0.1:8420` in a browser
      without running anything by hand first.
- [ ] **The deployment shows up with correct state in the UI/API**:
      ```
      Invoke-RestMethod http://127.0.0.1:8420/api/deployments
      ```
      Confirm the deployment you created in step 0 appears, with the
      **correct port** (this is the specific bug that was fixed — before
      this change, `port` came back as `null` after a restart).
- [ ] **The serving process itself is actually usable**, not just the
      container running:
      ```
      Invoke-RestMethod http://127.0.0.1:<port>/v1/models
      ```
      This should return a valid model list. If the container is "Up"
      but this fails, the container restarted but the serving process
      inside it may have crashed — check `docker logs <container name>`.

## 2. Cold-boot / GPU-not-ready sanity check (optional but recommended)

The retry logic in `gpu_monitor.poll_with_retry()` and
`deployment_manager.reconcile(retries=3, retry_delay=3.0)` is unit-tested
against mocked failures (see `backend/test_deployment_manager_phase1.py`),
but that's not the same as observing it against a real cold boot. If you
want to confirm this for real:

- [ ] Check `stdout`/logs from `run.py` right after the reboot (Task
      Scheduler can be configured to log output, or run
      `python run.py` manually once right after boot to watch it live).
- [ ] Look for the `[startup] GPU check OK: ...` line, and note whether
      it took more than one attempt (there's no explicit "attempt N"
      log line currently — if you want that visibility, it's a small
      follow-up, not blocking Phase 1).

## 3. If anything above fails

Stop here. Do not mark Phase 1 "done." Note exactly which checklist item
failed and the actual observed behavior (error message, `docker logs`
output, Task Scheduler's `LastTaskResult` code, etc.) before deciding on
a fix — don't re-attempt the same fix blind.

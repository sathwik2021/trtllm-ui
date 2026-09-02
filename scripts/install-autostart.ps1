<#
.SYNOPSIS
    Registers a Windows Scheduled Task that starts trtllm-ui's FastAPI
    backend (run.py) automatically when you log in.

.DESCRIPTION
    Chosen mechanism: Windows Scheduled Task, trigger = "At log on of any
    user", NOT a full Windows Service.

    Why a Scheduled Task instead of a service (nssm / pywin32):
      - No extra dependency (nssm) or packaging (pywin32 service wrapper)
        to install and keep working across Python upgrades.
      - Trivial to inspect/edit/remove via Task Scheduler GUI or
        Get-ScheduledTask, which matters for a single-user dev box.
      - Runs in the same user session as Docker Desktop / WSL2, which
        this app already depends on (it shells out to `docker` on PATH).

    Trade-off, stated explicitly (per Phase 1 requirement to justify this
    choice rather than silently default to one):
      - This task only starts AFTER an interactive user logs in. It will
        NOT start the FastAPI backend before login, and will NOT run on
        a headless/server boot with no interactive session.
      - Docker Desktop itself is expected to have its own "start on
        login" setting enabled independently (this script doesn't touch
        Docker Desktop's own settings) -- the container's own
        `--restart unless-stopped` policy is what brings the *container*
        back once the Docker daemon is up, independent of this task.
      - If you need the backend running before any user logs in (e.g. a
        headless always-on box), use a Windows Service instead (nssm or
        pywin32's service wrapper) -- not implemented here; flagging as
        an alternative if your use case changes.

.PARAMETER PythonExe
    Full path to the venv's python.exe that has requirements.txt installed.
    Defaults to ".venv\Scripts\python.exe" relative to the repo root.

.PARAMETER RepoRoot
    Full path to the trtllm-ui repo root (folder containing run.py).
    Defaults to the parent of this script's folder.

.EXAMPLE
    # Run from an elevated PowerShell prompt, from the repo root:
    .\scripts\install-autostart.ps1

.EXAMPLE
    .\scripts\install-autostart.ps1 -PythonExe "D:\trtllm-ui\.venv\Scripts\python.exe" -RepoRoot "D:\trtllm-ui"
#>

param(
    [string]$PythonExe = $(Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$RepoRoot  = $(Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$TaskName = "trtllm-ui-autostart"

$PythonExe = (Resolve-Path $PythonExe -ErrorAction SilentlyContinue).Path
$RepoRoot  = (Resolve-Path $RepoRoot -ErrorAction SilentlyContinue).Path

if (-not $PythonExe) {
    Write-Error "Could not resolve PythonExe path. Pass -PythonExe explicitly, e.g. -PythonExe 'D:\trtllm-ui\.venv\Scripts\python.exe'"
    exit 1
}
if (-not $RepoRoot -or -not (Test-Path (Join-Path $RepoRoot "run.py"))) {
    Write-Error "Could not find run.py under RepoRoot ('$RepoRoot'). Pass -RepoRoot explicitly."
    exit 1
}

Write-Host "Python exe : $PythonExe"
Write-Host "Repo root  : $RepoRoot"
Write-Host "Task name  : $TaskName"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Task '$TaskName' already exists -- removing it first so this script is safely re-runnable."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "run.py" -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Small delay so Docker Desktop / WSL2 has a moment to start; the app's
# own cold-boot retry logic (gpu_monitor.poll_with_retry, reconcile())
# handles the rest even if Docker isn't fully up yet.
$trigger.Delay = "PT30S"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)   # no time limit -- this is a long-running server process

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Auto-starts trtllm-ui's FastAPI backend (run.py) at user logon." | Out-Null

Write-Host ""
Write-Host "Installed. The task will run at your next logon."
Write-Host "To test right now without rebooting:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Then check it's actually listening:"
Write-Host "    Invoke-WebRequest http://127.0.0.1:8420 -UseBasicParsing"
Write-Host ""
Write-Host "To remove later: .\scripts\uninstall-autostart.ps1"

<#
.SYNOPSIS
    Removes the trtllm-ui auto-start Scheduled Task created by
    install-autostart.ps1.
#>

$ErrorActionPreference = "Stop"
$TaskName = "trtllm-ui-autostart"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Host "No task named '$TaskName' found -- nothing to do."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Removed scheduled task '$TaskName'."
Write-Host "Note: this does NOT stop the FastAPI process or any Docker containers currently running -- stop those separately if needed."

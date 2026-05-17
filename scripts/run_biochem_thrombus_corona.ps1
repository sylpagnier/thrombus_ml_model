# Biochem corrector with thrombus-corona preset (gelation prior gate + 3-hop graph corona,
# temporal COMSOL anchors, data-only step 2). From repo root:
#   .\scripts\run_biochem_thrombus_corona.ps1
# Optional: pass extra args to Python, e.g.
#   .\scripts\run_biochem_thrombus_corona.ps1 -ExtraArgs @("--epochs", "24")

param(
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_PRESET = "thrombus_corona"
if (-not $env:BIOCHEM_STOP_AFTER_TEACHER) {
    $env:BIOCHEM_STOP_AFTER_TEACHER = "0"
}

Write-Host "Repo: $RepoRoot"
Write-Host "BIOCHEM_PRESET=$env:BIOCHEM_PRESET"
python -m src.training.train_biochem_corrector @ExtraArgs

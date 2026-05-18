# EXPERIMENTAL / UNVALIDATED — thrombus-corona preset (gelation gate + 3-hop prior dilation,
# L_PhysTemp, teacher + corrector). Not recommended until μ teacher is stable on patient007
# (see src/docs/BIOCHEM_TRAINING_PROGRESS.md). Prefer run_biochem_mu_formulation_study.ps1 first.
#
# From repo root:
#   .\scripts\run_biochem_thrombus_corona.ps1
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

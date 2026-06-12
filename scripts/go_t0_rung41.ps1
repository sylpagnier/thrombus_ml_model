# Rung 4.1 rules-species ladder (CUDA only): diagnostics + eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung41.ps1"

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,27,53",
    [string] $RulesMode = "s0",
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

Write-Host "[NEW] Species teacher diagnostics (val split, CUDA)" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "go_t0_species_diagnostics.ps1") -Device cuda

Write-Host "[NEW] Rung 4.1 eval ($Anchor) mode=$RulesMode" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung41.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--rules-mode", $RulesMode
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung41 eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.1 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung1241.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--rules-mode", $RulesMode
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung1241 viz" -PyArgs $vizArgs

Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung41_eval.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung1241_$Anchor.png" -ForegroundColor Green

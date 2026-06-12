# Hybrid teacher-species rule sweep (baked anchors from dump_teacher_species_to_anchors.py).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_sweep_clot_hybrid_species.ps1"
#   powershell ... -Fast -Anchor patient007

param(
    [string] $AnchorDir = "outputs/biochem/anchors_teacher_species",
    [string] $Anchor = "",
    [switch] $Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$pyArgs = @(
    "scripts/sweep_clot_rule_architectures.py",
    "--hybrid-species",
    "--anchor-dir", $AnchorDir
)
if ($Anchor) { $pyArgs += @("--anchor", $Anchor) }
if ($Resume) { $pyArgs += "--resume" }

Write-Host ""
Write-Host "[NEW] hybrid teacher-species rule sweep" -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "hybrid species sweep" -PyArgs $pyArgs

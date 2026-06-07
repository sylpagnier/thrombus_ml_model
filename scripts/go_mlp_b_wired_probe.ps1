# Smoke scorecard for wired deploy (mlp_band commit inside vision band).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mlp_b_wired_probe.ps1 -Fast

param(
    [string[]] $Anchors = @("patient003", "patient007", "patient006"),
    [string] $ClotPhiCheckpoint = "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
    [switch] $Fast
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$TeacherCheckpoint = "outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth"
$TimeStride = if ($Fast) { 5 } else { 1 }

Write-Host "[NEW] B_wired mlp_band deploy probe (Lane A ckpts)" -ForegroundColor Cyan

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\run_mlp_clot_inject_probe.py"),
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint,
    "--anchors", ($Anchors -join ","),
    "--legs", "B_wired",
    "--time-stride", "$TimeStride",
    "--out", "outputs/biochem/mlp_clot_inject_probe/b_wired_smoke.json"
)
if ($Fast) { $pyArgs += "--fast" }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }
Write-Host "[OK]  outputs/biochem/mlp_clot_inject_probe/b_wired_smoke.json" -ForegroundColor Green

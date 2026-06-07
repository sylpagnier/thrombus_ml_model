# Step-1 deploy gate diagnostic (phi / mu_mlp / commit inside allowed vision).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_diagnose_deploy_gate.ps1
#   powershell ... -Leg B_deploy -Anchor patient007
#   powershell ... -Fast   # 3 rollout keyframes only

param(
    [string] $Anchor = "patient007",
    [ValidateSet("B_wired", "B_deploy", "B")]
    [string] $Leg = "B_wired",
    [switch] $Fast,
    [string] $TeacherCheckpoint = "outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth",
    [string] $ClotPhiCheckpoint = "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
    [string] $Out = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

Write-Host "[NEW] Deploy gate diagnose (step 1) leg=$Leg anchor=$Anchor" -ForegroundColor Cyan

$pyArgs = @(
    (Join-Path $RepoRoot "scripts\diagnose_deploy_gate.py"),
    "--anchor", $Anchor,
    "--leg", $Leg,
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--clot-phi-checkpoint", $ClotPhiCheckpoint
)
if ($Fast) { $pyArgs += "--fast" }
if ($Out) { $pyArgs += @("--out", $Out) }

$rc = Invoke-PythonRc @pyArgs
if ($rc -ne 0) { exit $rc }
Write-Host "[OK]  See outputs/biochem/diagnostics/deploy_gate.json" -ForegroundColor Green

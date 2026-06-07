# Clot baseline: one-command train + promote (Lane A = GNODE teacher + clot-phi MLP).
#
# Full train (~hours: mu unlock + dump + 35ep clot):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot.ps1
#
# Fast retrain (skip mu unlock if promoted teacher exists; reuse dump; 20ep clot):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot.ps1 -Fast
#
# Skip mu unlock but still dump+clot (e.g. after a false [ERR] from old Invoke-PythonRc):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot.ps1 -SkipMuUnlock
#
# Promote only (no training):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_baseline_clot.ps1 -PromoteOnly
#
# Predict smoke (needs manifest + anchor graph):
#   python -m src.inference --anchor patient007

param(
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $PromoteOnly,
    [switch] $SkipMuUnlock,
    [string] $TeacherCkpt = "",
    [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
    [string] $OutAnchorDir = "outputs\biochem\gnode10_sweep\anchors_gnode12_predkine_uvp",
    [int] $MuUnlockEpochs = 6,
    [int] $ClotEpochs = 35,
    [double] $MuRatioMax = 20,
    [switch] $SkipGate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$BaselineDir = Join-Path $RepoRoot "outputs\biochem\clot_baseline"
$PromotedTeacher = Join-Path $BaselineDir "teacher_best_high_mu.pth"

function Invoke-PromoteClotBaseline {
    $promoteArgs = @(
        "scripts/promote_clot_baseline.py",
        "--dump-dir", ($OutAnchorDir -replace '\\', '/'),
        "--eval-json", ("outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/multi_anchor.jsonl" -replace '\\', '/')
    )
    if ($TeacherCkpt) { $promoteArgs += @("--teacher", $TeacherCkpt) }
    Invoke-PythonRcCheck @promoteArgs -Label "promote_clot_baseline"
}

if ($PromoteOnly) {
    Write-Host "[NEW] promote clot baseline only" -ForegroundColor Cyan
    Invoke-PromoteClotBaseline
    exit 0
}

$clotEpochs = $ClotEpochs
$muUnlockEpochs = $MuUnlockEpochs
$skipMu = [bool]$SkipMuUnlock
$skipDump = $false
$teacherArg = $TeacherCkpt

if ($Fast) {
    $clotEpochs = 20
    if ((Test-Path $PromotedTeacher) -and -not $Fresh) {
        $skipMu = $true
        $teacherArg = "outputs/biochem/clot_baseline/teacher_best_high_mu.pth"
        Write-Host "[i]  Fast: skip mu unlock (promoted teacher)" -ForegroundColor DarkGray
    } else {
        $muUnlockEpochs = 3
        Write-Host "[i]  Fast: short mu unlock 3ep (no promoted teacher)" -ForegroundColor DarkGray
    }
    if ((Test-Path (Join-Path $RepoRoot $OutAnchorDir)) -and -not $Fresh) {
        $skipDump = $true
        Write-Host "[i]  Fast: reuse dump $OutAnchorDir" -ForegroundColor DarkGray
    }
}

$laneScript = Join-Path $PSScriptRoot "go_gnode12_lane_a.ps1"
$lanePsArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", $laneScript,
    "-JuneAnchorDir", $JuneAnchorDir,
    "-OutAnchorDir", $OutAnchorDir,
    "-ClotLeg", "gnode12_lane_a_clotphi",
    "-MuRatioMax", $MuRatioMax,
    "-ClotEpochs", $clotEpochs,
    "-MuUnlockEpochs", $muUnlockEpochs
)
if ($teacherArg) { $lanePsArgs += @("-TeacherCkpt", $teacherArg) }
if ($skipMu) { $lanePsArgs += "-SkipMuUnlock" }
if ($skipDump) { $lanePsArgs += "-SkipDump" }
if ($SkipGate) { $lanePsArgs += "-SkipGate" }

Write-Host "[NEW] clot baseline train (Lane A)" -ForegroundColor Cyan
& powershell @lanePsArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[NEW] promote stable deploy paths" -ForegroundColor Cyan
Invoke-PromoteClotBaseline

Write-Host "[OK]  baseline ready:" -ForegroundColor Green
Write-Host "  teacher: outputs/biochem/clot_baseline/teacher_best_high_mu.pth" -ForegroundColor DarkGray
Write-Host "  clot-phi: outputs/biochem/clot_baseline/clot_phi_best.pth" -ForegroundColor DarkGray
Write-Host "  manifest: outputs/biochem/clot_baseline/manifest.json" -ForegroundColor DarkGray
Write-Host "  predict: python -m src.inference --anchor patient007" -ForegroundColor DarkGray
Write-Host "  viz: .\scripts\go_baseline_clot_viz.ps1" -ForegroundColor DarkGray

exit 0

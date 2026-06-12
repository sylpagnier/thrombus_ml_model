# Star 1 (T1): train hybrid clot trigger on GT flow + GT species, then viz.
# Honest default: full-mesh loss, no GT mu band. Retrain required after pivot (-Fresh).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_trigger_t1.ps1" -Fresh
#   powershell ... -Fast -SkipTrain
#   powershell ... -VizOnly -Anchor patient007
#   powershell ... -OracleBand   # legacy debug only

param(
    [string] $Anchor = "patient007",
    [string] $Anchor2 = "patient002",
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $SkipTrain,
    [switch] $VizOnly,
    [switch] $OracleBand
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutDir = "outputs/biochem/clot_trigger/t1"
$Ckpt = "$OutDir/clot_trigger_t1_best.pth"
$VizDir = "outputs/biochem/viz/clot_trigger"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutDir) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if ($Fresh -and -not $VizOnly) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot "$OutDir\clot_trigger_t1_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot "$OutDir\clot_trigger_t1_train_log.jsonl")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RepoRoot "$OutDir\clot_phi_last.pth")
}

$env:PYTHONUNBUFFERED = "1"

if (-not $SkipTrain -and -not $VizOnly) {
    Write-Host "[NEW] T1 train hybrid trigger (GT flow + GT species, deploy rollout + ceiling loss)" -ForegroundColor Cyan
    $trainArgs = @("scripts/train_clot_trigger_t1.py")
    if ($Fast) { $trainArgs += "--fast" }
    if ($OracleBand) { $trainArgs += "--oracle-band" }
    Invoke-PythonRcCheck -Label "t1 train" -PyArgs $trainArgs
}

if (-not (Test-Path (Join-Path $RepoRoot $Ckpt))) {
    Write-Host "[WARN] missing ckpt $Ckpt -- train first or drop -SkipTrain" -ForegroundColor Yellow
    exit 1
}

foreach ($anc in @($Anchor, $Anchor2)) {
    if (-not $anc) { continue }
    Write-Host "[NEW] T1 viz $anc" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "t1 viz $anc" -PyArgs @(
        "scripts/viz_clot_trigger_t1.py",
        "--anchor", $anc,
        "--checkpoint", $Ckpt,
        "--out", "$VizDir/t1_$anc.png"
    )
}

Write-Host ""
Write-Host "[OK] ckpt -> $Ckpt" -ForegroundColor Green
Write-Host "[OK] viz  -> $VizDir/t1_$Anchor.png (and $Anchor2)" -ForegroundColor Green

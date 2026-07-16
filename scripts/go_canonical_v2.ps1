# Train WC_canonical_v2: WC_mat_3hop + SPECIES_DYNAMIC_OCCLUSION=1 (Pivot 3 folded in).
# Promotes the best checkpoint to mat_canonical_deploy/species/best.pth when -Promote is set.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_canonical_v2.ps1 -Fresh
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_canonical_v2.ps1 -Fresh -Promote
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_canonical_v2.ps1 -Fast        # smoke (10 ep)

param(
    [switch] $Fresh,
    [switch] $Fast,
    [switch] $Promote
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$Leg = "WC_canonical_v2"
$CkptPath = "outputs/biochem/biochem_gnn/mat_growth_ladder/$Leg/species/best.pth"
$PromotePath = "outputs/biochem/biochem_gnn/mat_canonical_deploy/species/best.pth"

# ---- train ----
$trainArgs = @("-m", "src.training.train_species_pushforward_continuous",
    "--leg", $Leg)
if ($Fresh) { $trainArgs += "--fresh" }
if ($Fast)  { $trainArgs += @("--epochs", "10", "--early-stop", "6", "--max-windows", "16") }

Write-Host "[i] Training $Leg ..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "train $Leg" -PyArgs $trainArgs

# ---- eval ----
Write-Host "[i] Evaluating $Leg ..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "eval $Leg" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--leg", $Leg,
    "--compare"
)

# ---- promote ----
if ($Promote) {
    Write-Host "[i] Promoting $CkptPath -> $PromotePath ..." -ForegroundColor Cyan
    $PromoteDir = Split-Path -Parent $PromotePath
    if (-not (Test-Path $PromoteDir)) { New-Item -ItemType Directory -Force -Path $PromoteDir | Out-Null }
    Copy-Item -Force $CkptPath $PromotePath
    Write-Host "[OK] Promoted to $PromotePath" -ForegroundColor Green
}

Write-Host "[OK] go_canonical_v2 done. ckpt=$CkptPath" -ForegroundColor Green

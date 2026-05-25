# K10g sanity: paint wall-adjacent mu_eff from COMSOL GT during rollout (proves viz scale + band).
# Uses existing teacher ckpt; no training. Requires K10e forward_policy in ckpt OR set K10E env below.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k10g_oracle_clots_viz.ps1"

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:BIOCHEM_MU_K10E_SIMPLE = "1"
$env:BIOCHEM_K10G_ORACLE_CLOTS = "1"
$env:BIOCHEM_MU_IC_STEADY_KIN = "1"
$env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_K10E_D_PEAK_ND = "0.008"
$env:BIOCHEM_K10E_SIGMA_ND = "0.008"
$env:BIOCHEM_K10E_SDF_MAX_ND = "0.04"

$Ckpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
if (-not (Test-Path $Ckpt)) {
    Write-Host "Missing $Ckpt — run K10f first." -ForegroundColor Red
    exit 1
}

python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint $Ckpt --patient patient007
exit $LASTEXITCODE

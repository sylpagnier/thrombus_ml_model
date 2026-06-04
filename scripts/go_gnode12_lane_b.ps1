# Rung 12 Lane B: gnode11_finish corrector rollout -> dump (pred kine) -> clot-phi.
#
# A/B vs Lane A (teacher + optional mu unlock): same mu_ratio_max and June anchor cache.
# Prereq: go_gnode11_finish.ps1 (corrector ckpt in archive or outputs/biochem/).
# Preflight MinGtPosFrac default 0.38 (not 0.55): corrector pred-flow widens dgamma mask vs Lane A (~0.80).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_b.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_b.ps1 -CorrectorCkpt outputs\biochem\biochem_best_high_mu.pth
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gnode12_lane_b.ps1 -SkipDump -SkipGate
#
# After run:
#   python scripts/check_gnode12_lane_b_gate.py

param(
    [string] $CorrectorCkpt = "",
    [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
    [string] $OutAnchorDir = "outputs\biochem\gnode10_sweep\anchors_gnode12_corrector_predkine_uvp",
    [string] $ClotLeg = "gnode12_lane_b_clotphi",
    [double] $MuRatioMax = 20,
    [int] $ClotEpochs = 35,
    [double] $MinGtPosFrac = 0.38,
    [double] $MinClotMinF1 = 0.26,
    [switch] $SkipDump,
    [switch] $SkipClot,
    [switch] $SkipViz,
    [switch] $SkipGate,
    [switch] $SkipLaneACompare
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_gnode12_env.ps1")

$CorrectorPath = Resolve-Gnode12CorrectorCkpt -UserPath $CorrectorCkpt
if (-not $CorrectorPath) {
    Write-Host "[ERR] Corrector ckpt missing. Run go_gnode11_finish.ps1 or pass -CorrectorCkpt." -ForegroundColor Red
    exit 1
}

$EvalJson = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\multi_anchor.jsonl"

Write-Host "[NEW] GNODE 12 Lane B (corrector dump + clot-phi)" -ForegroundColor Cyan
Write-Host "[i]  corrector=$CorrectorPath" -ForegroundColor DarkGray
Write-Host "[i]  mu_ratio_max=$MuRatioMax" -ForegroundColor DarkGray
Write-Host "[i]  dump -> $OutAnchorDir | clot=$ClotLeg" -ForegroundColor DarkGray

Invoke-Gnode12DumpClotLeg `
    -RolloutCkptPath $CorrectorPath `
    -JuneAnchorDir $JuneAnchorDir `
    -OutAnchorDir $OutAnchorDir `
    -ClotLeg $ClotLeg `
    -MuRatioMax $MuRatioMax `
    -ClotEpochs $ClotEpochs `
    -MinGtPosFrac $MinGtPosFrac `
    -SkipDump:$SkipDump `
    -SkipClot:$SkipClot `
    -SkipViz:$SkipViz `
    -LaneLabel "B" `
    -BaselineNote "(compare Lane A p007 ~0.750)"

if ($SkipClot) {
    if (-not $SkipGate) {
        Write-Host "[WARN] clot skipped; lane gate needs eval json from a full run" -ForegroundColor Yellow
    }
    exit 0
}

if (-not $SkipGate) {
    Write-Host "[NEW] lane B gate check" -ForegroundColor Cyan
    $gateArgs = @(
        "scripts/check_gnode12_lane_b_gate.py",
        "--eval-json", $EvalJson,
        "--min-clot-min-f1", "$MinClotMinF1"
    )
    if ($SkipLaneACompare) { $gateArgs += "--skip-lane-a-compare" }
    $gateRc = Invoke-PythonRc @gateArgs
    if ($gateRc -ne 0) { exit $gateRc }
}

Write-Host "[OK]  GNODE 12 Lane B complete." -ForegroundColor Green

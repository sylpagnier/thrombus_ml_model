param(
    [int]    $Epochs        = 14,
    [int]    $EarlyStop     = 14,
    [double] $Lr            = 5e-5,
    [string] $ValAnchor     = "patient041",
    [string] $TrainAnchors  = "",
    [double] $GatePct       = 25,
    [string] $RunRoot       = "outputs/biochem/eda/phase1",
    [string] $InitCkpt      = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth",
    [string] $Leg           = "WG_phase1_baseline",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit
)

# Phase 1 re-baseline (docs/WALL_MODEL_PLAN.md s21.3, s22).
#
# Deliberately NOT go_wg_stenosis_subcohort_ft.ps1. That script encodes the 5-vessel stenosis
# study's invariants -- a 2-8 vessel range check, its own forbidden list, patient043 as the
# default holdout -- every one of which is wrong here. Phase 1 trains on cohort v2 (26 vessels),
# seals 8, and must never touch patient043. Bolting those semantics onto that script would have
# meant loosening exactly the guards that make it trustworthy for its own legs.
#
# This leg is a REFERENCE POINT, not an A/B. It switches the whole Phase-0 foundation on at
# once -- cohort v2 + analytic priors + rel_max labels + the measured mass window -- so nothing
# is attributable across it and no number from sections 9-20 is comparable with anything after
# it. Every later leg is single-variable against THIS.
#
#   .\scripts\go_phase1_baseline.ps1 -Fresh                       # 14 epochs, full run
#   .\scripts\go_phase1_baseline.ps1 -Epochs 1 -EarlyStop 1 -Fresh -SkipViz   # smoke

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# --- resolve the sealed set and the train cohort from the single source of truth -------------
$SealedCsv = (python -c @"
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_GENERALIZATION
print(','.join(WALL_COHORT_V2_GENERALIZATION))
"@).Trim()
if (-not $SealedCsv) { Write-Host "[ERR] could not resolve the sealed set" -ForegroundColor Red; exit 1 }
$Sealed = @($SealedCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# The sealed set is spent ONCE. Using one as the val anchor tunes selection against it every
# epoch, which silently converts a generalization claim into a development number.
if ($Sealed -contains $ValAnchor) {
    Write-Host "[ERR] $ValAnchor is SEALED: $SealedCsv" -ForegroundColor Red
    Write-Host "[ERR] Using it as -ValAnchor would spend the seal on epoch selection." -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrWhiteSpace($TrainAnchors)) {
    $TrainAnchors = (python -c @"
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN
print(','.join(a for a in WALL_COHORT_V2_TRAIN if a != '$ValAnchor'))
"@).Trim()
}
if (-not $TrainAnchors) { Write-Host "[ERR] could not resolve cohort v2 train set" -ForegroundColor Red; exit 1 }
$TrainList = @($TrainAnchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

foreach ($bad in $Sealed) {
    if ($TrainList -contains $bad) {
        Write-Host "[ERR] SEAL BREACH: $bad is both sealed and in the train list" -ForegroundColor Red
        exit 1
    }
}
foreach ($junk in @("patient002", "patient023")) {
    if ($TrainList -contains $junk) {
        Write-Host "[ERR] $junk is a standing data-quality exclusion (see mat_growth_simple.py)" -ForegroundColor Red
        exit 1
    }
}
if ($TrainList -contains $ValAnchor) {
    Write-Host "[ERR] val anchor $ValAnchor must not appear in the train list" -ForegroundColor Red
    exit 1
}
if ($TrainList.Count -lt 15) {
    Write-Host "[ERR] Phase 1 expects the full cohort v2 (~25 train vessels), got $($TrainList.Count)" -ForegroundColor Red
    Write-Host "[i] A short list usually means the constant was edited or -TrainAnchors was passed by mistake." -ForegroundColor DarkGray
    exit 1
}

$OutDir  = Join-Path $RepoRoot $RunRoot
$ArmDir  = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_val_cold.json"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

Write-Host "[NEW] PHASE 1 RE-BASELINE ($Leg): $Epochs ep / ES $EarlyStop / lr=$Lr" -ForegroundColor Cyan
Write-Host "[i] This is a REFERENCE POINT, not an A/B. Nothing is attributable across it." -ForegroundColor Yellow
Write-Host "[i]   cohort    5 ad-hoc -> cohort v2 ($($TrainList.Count) train, $($Sealed.Count) sealed)" -ForegroundColor DarkGray
Write-Host "[i]   priors    stored (leaked CFD) -> analytic   (s16.1 / s17 Z2 contract)" -ForegroundColor DarkGray
Write-Host "[i]   labels    fixed 1e-4 -> rel_max @ 10% of each vessel's peak Mat  (s20.3)" -ForegroundColor DarkGray
Write-Host "[i]   selection mass window [0.5,1.5] -> [1.2,4.5], target 3.0  (s20.2 measured 3.04x)" -ForegroundColor DarkGray
Write-Host "[i] train($($TrainList.Count))=$TrainAnchors" -ForegroundColor DarkGray
Write-Host "[i] val=$ValAnchor (dev cohort, NOT sealed)   gate pct=$GatePct" -ForegroundColor DarkGray
Write-Host "[i] SEALED, never touched: $SealedCsv" -ForegroundColor DarkGray
Write-Host "[i] VERIFY IN THE LOG: prior source analytic, 'latent'/label scale bound, and that" -ForegroundColor DarkGray
Write-Host "[i]   the pack cache path carries the _prior-analytic tag (else it reused leaked packs)." -ForegroundColor DarkGray

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, (Join-Path $ArmDir "train_log.jsonl"), (Join-Path $ArmDir "best_salvage.pth") -ErrorAction SilentlyContinue
}

if (-not $EvalOnly) {
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", $Leg,
        "--out", $ArmCkpt,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--lr", "$Lr",
        "--anchors", $TrainAnchors,
        "--val-anchor", $ValAnchor,
        "--exclude-val-from-train",
        "--drop-xy",
        "--deploy-freq", "1"
    )
    if ($NoInit) { $trainArgs += "--no-init" }
    else { $trainArgs += @("--init", $InitCkpt, "--init-mode", "full") }

    $null = Invoke-PythonRcCheck -Label "$Leg train" -PyArgs $trainArgs

    if (-not (Test-Path $ArmCkpt)) {
        Write-Host "[ERR] $Leg produced no checkpoint (not even a salvage promotion)" -ForegroundColor Red
        exit 1
    }
}

$null = Invoke-PythonRcCheck -Label "$Leg val eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $ArmCkpt,
    "--no-baseline",
    "--anchors", $ValAnchor,
    "--pocket-gate-pct", "$GatePct",
    "--out", $ArmHold
)

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] Grade with: python scripts/diag_leg_alignment.py --logs $ArmDir/train_log.jsonl" -ForegroundColor DarkGray
Write-Host "[i] The sealed set stays UNSPENT until an arm is chosen. Do not evaluate it now." -ForegroundColor Yellow

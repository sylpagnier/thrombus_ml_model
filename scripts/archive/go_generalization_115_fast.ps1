# Tier 1.15 -- fast generalization probe (single fold, NOT full LOVO).
#
# Question: if we train the growth specialist on clot-rich vessels that are
# geometrically similar to family_validation (holding family_validation OUT),
# does the compound deploy score generalize to those held-out vessels?
#
# Why the growth specialist and not a wall retrain: on this 4 GB GPU a single
# deploy-faithful coupled rollout is ~25-30 min/anchor, so a full WC_v7 wall
# retrain is ~hours/epoch and a leave-one-vessel-out (LOVO) sweep is tens of
# GPU-hours -- far past the 2 h "fast" bar. The growth trainer uses cheap
# tile/window rollouts and its per-epoch --compound-val prints the
# family_validation generalization read directly, so we get the signal in ~1-2 h.
#
# Train anchors : clot-rich, full-length vessels inside the sealed train set
#                 (001,005,006,007,010,013,016,020,029).
# Held out      : family_validation = 021,035,037 (val) ; challenge = 032.
# Wall backbone : locked WC_v7 (frozen; this probe only retrains growth).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_generalization_115_fast.ps1 -Fresh

param(
    [int]    $Epochs      = 12,
    [int]    $EarlyStop   = 5,
    [int]    $MaxWindows  = 36,
    [int]    $HopsK       = 5,
    [double] $LumenShapeWeight = 4.0,
    [int]    $FnW         = 6,
    [double] $FpW         = 2.5,
    [double] $Underpred   = 4.0,
    [string] $TrainAnchors = "patient001,patient005,patient006,patient007,patient010,patient013,patient016,patient020,patient029",
    [string] $ValAnchors   = "patient021,patient035,patient037",
    [string] $ValAnchor    = "patient021",
    [string] $WallCkpt   = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot    = "outputs/biochem/eda/gen_115_growth",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Ckpt = Join-Path $OutDir "growth_115.pth"
if ($Fresh) { Remove-Item -Force $Ckpt, (Join-Path $OutDir "best.json"), (Join-Path $OutDir "train_log.jsonl") -ErrorAction SilentlyContinue }

# FN / missed-mass tilt (Tier 2 lever): push lumen recall on the held-out family.
$env:SPECIES_LUMEN_SHAPE_FN_W = "$FnW"
$env:SPECIES_LUMEN_SHAPE_FP_W = "$FpW"
$env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "$Underpred"

Write-Host "[NEW] gen_115 growth probe epochs=$Epochs val=$ValAnchors fn_w=$FnW underpred=$Underpred" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors" -ForegroundColor DarkGray

$trainArgs = @(
    "-m", "src.training.train_offwall_growth",
    "--anchors", $TrainAnchors,
    "--val-anchors", $ValAnchors,
    "--val-anchor", $ValAnchor,
    "--epochs", "$Epochs",
    "--early-stop", "$EarlyStop",
    "--max-windows", "$MaxWindows",
    "--hops-k", "$HopsK",
    "--supervise-mode", "frontier_ge2",
    "--frontier-hops", "2",
    "--loss-mode", "loss_lumen_shape",
    "--lumen-shape-weight", "$LumenShapeWeight",
    "--ckpt-metric", "compound_primary",
    "--train-feat-source", "band",
    "--mat-leg", "WC_v7_clot_phi_mse",
    "--init", $WallCkpt,
    "--compound-val",
    "--compound-val-route", "frontier_offwall",
    "--compound-val-frontier-hops", "0.5",
    "--wall-ckpt", $WallCkpt,
    "--wall-clot-floor-delta", "0.10",
    "--compound-val-every", "2",
    "--out", $Ckpt
)
$null = Invoke-PythonRcCheck -Label "gen_115 growth train" -PyArgs $trainArgs

Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
Write-Host "[OK] gen_115 growth done -> $Ckpt" -ForegroundColor Green
Write-Host "[i] read per-epoch 'compound-val patientNNN' lines for the family_validation generalization curve" -ForegroundColor DarkGray

# Confirmation run: fi_mat_FG (FI+Mat+fibrinogen) vs the locked baseline.
#
# The locked model IS the FI+Mat reference (its canonical deployed best), so we only
# train one new leg (fi_mat_FG), then eval both under the identical deploy_frozen harness
# and produce a head-to-head guiding-metric verdict. Budget: ~1h (single training leg).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_fimat_fg_confirm.ps1
#   powershell ... -Epochs 32 -MaxWindows 96 -Fresh

param(
    [int] $Epochs = 36,
    [int] $EarlyStop = 12,
    [int] $MaxWindows = 100,
    [double] $Lr = 3e-4,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $Fresh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/fimat_fg_confirm"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
$BaselineManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_baseline.json"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }
if (-not (Test-Path $BaselineManifest)) { throw "missing baseline manifest: $BaselineManifest" }

$Label = "fi_mat_FG"
$Channels = @(8, 11, 7)
$chStr = ($Channels | ForEach-Object { "$_" }) -join ","

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

# ----------------------------------------------------------------------------
# Train the candidate (fi_mat_FG) + eval deploy_frozen
# ----------------------------------------------------------------------------
$legDir = Join-Path $RunRoot $Label
$speciesDir = Join-Path $legDir "species"
$evalDir = Join-Path $legDir "eval"
New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
$speciesOut = Join-Path $speciesDir "best.pth"
$evalOut = Join-Path $evalDir "deploy_ab_eval.json"
$manifest = Join-Path $legDir "manifest.json"

if ($Fresh) {
    if (Test-Path $speciesOut) { Remove-Item $speciesOut -Force }
    if (Test-Path $evalOut) { Remove-Item $evalOut -Force }
}

Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE -ErrorAction SilentlyContinue
$env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $chStr
$env:SPECIES_PUSHFORWARD_ARCH = "sage"
python -c "from src.biochem_gnn.config import apply_train_recipe_env; apply_train_recipe_env()" | Out-Null
$env:SPECIES_TRAIN_VEL_SOURCE = "gt"
$env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
$env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
$env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
$env:SPECIES_ROLLOUT_IC_SOURCE = "resting"

Write-Host "[run] [$Label] channels=$chStr ($Epochs ep, early-stop=$EarlyStop, max_windows=$MaxWindows)" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "[$Label] train" -PyArgs @(
    "-m", "src.training.train_species_pushforward_continuous",
    "--phase", "biochem_gnn",
    "--anchors", $Anchors,
    "--val-anchor", "patient007",
    "--epochs", "$Epochs",
    "--early-stop", "$EarlyStop",
    "--max-windows", "$MaxWindows",
    "--unroll", "10",
    "--lr", "$Lr",
    "--arch", "sage",
    "--init", $InitWarm,
    "--out", $speciesOut
)

$payload = @{
    name = "biochem_gnn_fimat_fg_confirm"
    version = 1
    baseline = @{
        species_gnn_ckpt = (RelPath $speciesOut)
        viscosity_beta = (RelPath $BetaCkpt)
        kinematics_ckpt = "outputs/kinematics/kinematics_best.pth"
        train_val_anchor = "patient007"
        flow_modes = "kinematics"
        gamma_mode = "max"
        deploy_horizon = "full"
        clot_score = "guiding"
        pushforward_arch = "sage"
        species_channels = @($Channels)
        loao_auto = "0"
    }
}
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)

$env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $chStr
Write-Host "[run] [$Label] eval deploy_frozen" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "[$Label] eval" -PyArgs @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--manifest", $manifest,
    "--modes", "deploy_frozen",
    "--times", "53,200",
    "--anchors", $Anchors,
    "--out", $evalOut
)

# ----------------------------------------------------------------------------
# Eval the locked baseline (current best, FI+Mat) under identical harness
# ----------------------------------------------------------------------------
$baseEval = Join-Path $RunRoot "baseline_locked/eval/deploy_ab_eval.json"
New-Item -ItemType Directory -Force -Path (Split-Path $baseEval) | Out-Null
Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE -ErrorAction SilentlyContinue
Write-Host "[run] [baseline_locked] eval deploy_frozen (current best)" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "[baseline_locked] eval" -PyArgs @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--manifest", $BaselineManifest,
    "--modes", "deploy_frozen",
    "--times", "53,200",
    "--anchors", $Anchors,
    "--out", $baseEval
)

# ----------------------------------------------------------------------------
# Summarize: fi_mat_FG vs locked baseline (guiding rank metric)
# ----------------------------------------------------------------------------
$screenManifest = @{ legs = @(@{ label = $Label; channels = @($Channels); addon_channel = 7; eval = $evalOut });
                     epochs = $Epochs; max_windows = $MaxWindows }
$manifestPath = Join-Path $RunRoot "screen_manifest.json"
[System.IO.File]::WriteAllText($manifestPath, ($screenManifest | ConvertTo-Json -Depth 6), $utf8NoBom)

$SummaryJson = Join-Path $RunRoot "fimat_fg_confirm_summary.json"
$SummaryMd = Join-Path $RunRoot "fimat_fg_confirm_report.md"
$null = Invoke-PythonRcCheck -Label "fimat_fg confirm summary" -PyArgs @(
    "scripts/summarize_species_rank_ladder.py",
    "--screen-root", $RunRoot,
    "--baseline-eval", $baseEval,
    "--out-json", $SummaryJson,
    "--out-md", $SummaryMd,
    "--top-n", "1"
)

# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
$summary = Get-Content $SummaryJson -Raw | ConvertFrom-Json
$base = $summary.baseline_fi_mat
$cand = $summary.screen_ranked[0]
if ($cand -and $base) {
    $cm = $cand.metrics
    Write-Host ""
    Write-Host ("[i] locked baseline : p007 guiding {0:N3} | holdout {1:N3} | rank {2:N3}" -f $base.p007_guiding, $base.holdout_mean_guiding, $base.rank_score) -ForegroundColor Cyan
    Write-Host ("[i] fi_mat_FG        : p007 guiding {0:N3} | holdout {1:N3} | rank {2:N3}" -f $cm.p007_guiding, $cm.holdout_mean_guiding, $cm.rank_score) -ForegroundColor Cyan
    if ([double]$cm.rank_score -gt [double]$base.rank_score) {
        $delta = [double]$cm.rank_score - [double]$base.rank_score
        Write-Host ("[OK] CONFIRMED: fi_mat_FG beats locked baseline on guiding rank (delta +{0:N3})" -f $delta) -ForegroundColor Green
    } else {
        $delta = [double]$base.rank_score - [double]$cm.rank_score
        Write-Host ("[WARN] NOT CONFIRMED: fi_mat_FG does not beat locked baseline (behind by {0:N3})" -f $delta) -ForegroundColor Yellow
    }
}
Write-Host "[OK] report  -> $SummaryMd" -ForegroundColor Green

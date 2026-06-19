# 6h GPU sweep: maximize relaxed clot precision for fi_mat_FG_APR_thrombin while
# still predicting some true clots (recall floor).
#
# Levers: FP weight, speed-FP weight, recall floor, clot-weighted checkpoint selection
# with SPECIES_CONTINUOUS_CLOUT_SCORE=relaxed_prec_floor.
#
# Phase A: coarse grid (short).  Phase B: refine top-2 (longer).  Phase C: final long
# run on the overall best config, promoted to outputs/.../precision_best/.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_precision_6h.ps1
#   powershell ... -Floor 0.25 -Fresh

param(
    [double] $Floor = 0.30,
    [double] $WP007 = 0.5,
    [int] $CoarseEpochs = 16,
    [int] $CoarseWindows = 60,
    [int] $RefineEpochs = 34,
    [int] $RefineWindows = 100,
    [int] $FinalEpochs = 55,
    [int] $FinalWindows = 130,
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

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/precision_6h"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
if ($Fresh) { Get-ChildItem $RunRoot -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue }

$InitWarm = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
$BetaCkpt = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
if (-not (Test-Path $InitWarm)) { throw "missing init ckpt: $InitWarm" }
if (-not (Test-Path $BetaCkpt)) { throw "missing beta ckpt: $BetaCkpt" }

$Channels = "8,11,7,2,5"   # fi_mat + FG + APR + thrombin (canonical order)
$Deadline = (Get-Date).AddHours(6)
$BaselineManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_baseline.json"

function RelPath([string]$AbsPath) {
    return (Resolve-Path $AbsPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
}

function Train-Leg([hashtable]$Cfg) {
    $label = $Cfg.label
    $legDir = Join-Path $RunRoot $label
    $speciesDir = Join-Path $legDir "species"
    $evalDir = Join-Path $legDir "eval"
    New-Item -ItemType Directory -Force -Path $speciesDir, $evalDir | Out-Null
    $speciesOut = Join-Path $speciesDir "best.pth"
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $manifest = Join-Path $legDir "manifest.json"

    if (Test-Path $evalOut) {
        Write-Host "[skip] [$label] already evaluated" -ForegroundColor DarkYellow
        return $evalOut
    }

    # Base recipe first (only fills unset vars), then precision overrides.
    Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE -ErrorAction SilentlyContinue
    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $Channels
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    python -c "from src.biochem_gnn.config import apply_train_recipe_env; apply_train_recipe_env()" | Out-Null
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
    # Precision-targeted selection + FP penalties.
    $env:SPECIES_CONTINUOUS_CLOUT_SCORE = "relaxed_prec_floor"
    $env:SPECIES_CLOUT_PREC_REC_FLOOR = "$($Cfg.floor)"
    $env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "$($Cfg.clout_w)"
    $env:SPECIES_CONTINUOUS_FP_WEIGHT = "$($Cfg.fp_weight)"
    $env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT = "$($Cfg.speed_fp)"

    Write-Host "[run] [$label] fp=$($Cfg.fp_weight) speed_fp=$($Cfg.speed_fp) floor=$($Cfg.floor) clout_w=$($Cfg.clout_w) ep=$($Cfg.epochs) win=$($Cfg.windows)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$label] train" -PyArgs @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--anchors", $Anchors,
        "--val-anchor", "patient007",
        "--epochs", "$($Cfg.epochs)",
        "--early-stop", "$($Cfg.early_stop)",
        "--max-windows", "$($Cfg.windows)",
        "--unroll", "10",
        "--lr", "$Lr",
        "--arch", "sage",
        "--init", $InitWarm,
        "--out", $speciesOut
    )

    $payload = @{
        name = "biochem_gnn_precision_$label"
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
            species_channels = @(8, 11, 7, 2, 5)
            loao_auto = "0"
        }
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($manifest, ($payload | ConvertTo-Json -Depth 6), $utf8NoBom)
    [System.IO.File]::WriteAllText((Join-Path $legDir "leg_config.json"), ($Cfg | ConvertTo-Json -Depth 4), $utf8NoBom)

    $env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS = $Channels
    Write-Host "[run] [$label] eval deploy_frozen" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$label] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $manifest,
        "--modes", "deploy_frozen",
        "--times", "53,200",
        "--anchors", $Anchors,
        "--out", $evalOut
    )
    return $evalOut
}

function Summarize([string]$Tag) {
    $sj = Join-Path $RunRoot "precision_${Tag}_summary.json"
    $sm = Join-Path $RunRoot "precision_${Tag}_report.md"
    $null = Invoke-PythonRcCheck -Label "summary $Tag" -PyArgs @(
        "scripts/summarize_species_precision_sweep.py",
        "--sweep-root", $RunRoot,
        "--floor", "$Floor",
        "--w-p007", "$WP007",
        "--out-json", $sj,
        "--out-md", $sm
    )
    return $sj
}

# ----------------------------------------------------------------------------
# Phase 0: evaluate current best (locked) baseline under identical metric
# ----------------------------------------------------------------------------
$baseEval = Join-Path $RunRoot "baseline_locked/eval/deploy_ab_eval.json"
if ((Test-Path $BaselineManifest) -and -not (Test-Path $baseEval)) {
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
}

# ----------------------------------------------------------------------------
# Phase A: coarse grid
# ----------------------------------------------------------------------------
Write-Host "[i] Phase A coarse grid (deadline $Deadline)" -ForegroundColor DarkGray
$coarse = @(
    @{ label = "A_fp8_sp4";   fp_weight = 8;  speed_fp = 4;  floor = $Floor; clout_w = 0.8 }
    @{ label = "A_fp16_sp4";  fp_weight = 16; speed_fp = 4;  floor = $Floor; clout_w = 0.8 }
    @{ label = "A_fp16_sp10"; fp_weight = 16; speed_fp = 10; floor = $Floor; clout_w = 0.8 }
    @{ label = "A_fp32_sp10"; fp_weight = 32; speed_fp = 10; floor = $Floor; clout_w = 0.8 }
    @{ label = "A_fp32_sp16"; fp_weight = 32; speed_fp = 16; floor = $Floor; clout_w = 0.9 }
    @{ label = "A_fp48_sp16"; fp_weight = 48; speed_fp = 16; floor = $Floor; clout_w = 0.9 }
)
foreach ($c in $coarse) {
    $c.epochs = $CoarseEpochs; $c.early_stop = 8; $c.windows = $CoarseWindows
    if ((Get-Date) -gt $Deadline) { Write-Host "[WARN] deadline hit, stopping Phase A" -ForegroundColor Yellow; break }
    $null = Train-Leg $c
}
$sa = Summarize "phaseA"

# ----------------------------------------------------------------------------
# Phase B: refine the top-2 coarse configs (longer + precision tweaks)
# ----------------------------------------------------------------------------
$topCfg = (Get-Content $sa -Raw | ConvertFrom-Json).legs
$refine = @()
$rank = 0
foreach ($leg in $topCfg) {
    if ($rank -ge 2) { break }
    $cfg = $leg.config
    if (-not $cfg) { continue }
    $rank++
    # variant 1: longer at same knobs
    $refine += @{ label = "B${rank}_long"; fp_weight = $cfg.fp_weight; speed_fp = $cfg.speed_fp; floor = $Floor; clout_w = 0.85 }
    # variant 2: push precision harder (1.5x FP weight)
    $refine += @{ label = "B${rank}_fpup"; fp_weight = [int]([math]::Round($cfg.fp_weight * 1.5)); speed_fp = $cfg.speed_fp; floor = $Floor; clout_w = 0.9 }
}
Write-Host "[i] Phase B refine ($($refine.Count) legs)" -ForegroundColor DarkGray
foreach ($c in $refine) {
    $c.epochs = $RefineEpochs; $c.early_stop = 12; $c.windows = $RefineWindows
    if ((Get-Date) -gt $Deadline) { Write-Host "[WARN] deadline hit, stopping Phase B" -ForegroundColor Yellow; break }
    $null = Train-Leg $c
}
$sb = Summarize "phaseB"

# ----------------------------------------------------------------------------
# Phase C: final long run on overall best config
# ----------------------------------------------------------------------------
$best = (Get-Content $sb -Raw | ConvertFrom-Json).best
if ($best -and $best.config -and ((Get-Date) -lt $Deadline)) {
    $bc = $best.config
    $final = @{ label = "C_final"; fp_weight = $bc.fp_weight; speed_fp = $bc.speed_fp; floor = $Floor; clout_w = 0.9; epochs = $FinalEpochs; early_stop = 16; windows = $FinalWindows }
    Write-Host "[i] Phase C final long run from best=$($best.label)" -ForegroundColor DarkGray
    $null = Train-Leg $final
}
$sc = Summarize "final"

# ----------------------------------------------------------------------------
# Promote overall best ckpt (guarded: only if it beats the locked baseline)
# ----------------------------------------------------------------------------
$finalSummary = Get-Content $sc -Raw | ConvertFrom-Json
$finalBest = $finalSummary.best
$promoteOk = [bool]$finalSummary.promote_ok
if ($finalBest -and $promoteOk) {
    $promoteDir = Join-Path $RunRoot "precision_best"
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    $srcCkpt = Join-Path $RunRoot "$($finalBest.label)/species/best.pth"
    $srcManifest = Join-Path $RunRoot "$($finalBest.label)/manifest.json"
    if (Test-Path $srcCkpt) { Copy-Item $srcCkpt (Join-Path $promoteDir "species_gnn_best.pth") -Force }
    if (Test-Path $srcManifest) { Copy-Item $srcManifest (Join-Path $promoteDir "manifest.json") -Force }
    $finalBest | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $promoteDir "best_summary.json")
    Write-Host "[OK] PROMOTE: best=$($finalBest.label) beats locked baseline -> $promoteDir" -ForegroundColor Green
    Write-Host ("[OK] best p007 relaxed precision {0:N3} @ recall {1:N3}" -f $finalBest.metrics.p007_relaxed_prec, $finalBest.metrics.p007_relaxed_rec) -ForegroundColor Green
} elseif ($finalBest) {
    Write-Host "[WARN] NO PROMOTE: best sweep leg does not beat locked baseline; keeping current best." -ForegroundColor Yellow
}
Write-Host "[OK] precision 6h sweep complete: $RunRoot" -ForegroundColor Green

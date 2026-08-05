param(
    [int]    $Epochs         = 20,
    [int]    $EarlyStop      = 8,
    [double] $Lr             = 5e-5,
    [string] $HoldoutAnchor  = "patient020",
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_clotrich_nplus_v2",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_featfix/WG_featfix_03/best.pth",
    [string] $Leg            = "WG_clotrich_nplus_v2",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit,
    [switch] $NoFreeze
)

# Clot-rich N+ v2: mass-gated selection + deploy_horizon aux + light heads-only FT.
# Same featfix_03 stack / clot-rich LOAO (no 023/002). Holdout patient020.
#   .\scripts\go_wg_clotrich_nplus.ps1 -Epochs 20 -EarlyStop 8 -Fresh

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
$ArmDir = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
$ArmLog = Join-Path $ArmDir "train_log.jsonl"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

$trainCsv = (python -c @"
from src.biochem_gnn.mat_growth_simple import wall_gen_clot_rich_train_anchors
print(','.join(wall_gen_clot_rich_train_anchors(holdout='$HoldoutAnchor')))
"@).Trim()
if (-not $trainCsv -or $LASTEXITCODE -ne 0) {
    Write-Host "[ERR] failed to resolve clot-rich train anchors (holdout=$HoldoutAnchor)" -ForegroundColor Red
    exit 1
}
$trainList = @($trainCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$forbidden = @("patient002", "patient023", $HoldoutAnchor)
foreach ($bad in $forbidden) {
    if ($trainList -contains $bad) {
        Write-Host "[ERR] train list must not include $bad" -ForegroundColor Red
        Write-Host "[i] train=$trainCsv" -ForegroundColor DarkGray
        exit 1
    }
}
if ($trainList.Count -lt 10) {
    Write-Host "[ERR] expected >=10 clot-rich train vessels, got $($trainList.Count)" -ForegroundColor Red
    exit 1
}

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing featfix_03 warm-start ckpt: $InitCkpt" -ForegroundColor Red
    Write-Host "[i] Run go_wg_featfix_sweep.ps1 arm WG_featfix_03 first." -ForegroundColor DarkGray
    exit 1
}

$featfixTrainLeak = @("patient005", "patient006", "patient010", "patient023", "patient002")
if (-not $NoInit -and ($featfixTrainLeak -contains $HoldoutAnchor)) {
    Write-Host "[ERR] holdout=$HoldoutAnchor was in featfix_03 train set; warm-start would leak." -ForegroundColor Red
    Write-Host "[i] Use -NoInit, or hold out a vessel never seen by $InitCkpt (e.g. patient020)." -ForegroundColor DarkGray
    exit 1
}

Write-Host "[NEW] wg_clotrich_nplus ($Leg): $Epochs ep / ES $EarlyStop / lr=$Lr" -ForegroundColor Cyan
Write-Host "[i] v2 fixes=mass-gated select + deploy_horizon aux + mature_fp_on + freeze heads-only FT" -ForegroundColor DarkGray
Write-Host "[i] stack=featfix_03 (geom+flux, drop-xy, auto+coupled)" -ForegroundColor DarkGray
Write-Host "[i] train($($trainList.Count))=$trainCsv" -ForegroundColor DarkGray
Write-Host "[i] val/holdout=$HoldoutAnchor (deploy-faithful cold gate)" -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random (--NoInit)" -ForegroundColor DarkGray
} else {
    Write-Host "[i] init=$InitCkpt" -ForegroundColor DarkGray
}
Write-Host "[i] out=$ArmDir" -ForegroundColor DarkGray
Write-Host "[i] success bar: cold $HoldoutAnchor score > featfix_03 (~0.329) without spray (mass~1-1.5); target >=0.45" -ForegroundColor DarkGray

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host "[skip] $Leg already completed (eval JSON exists); pass -Fresh to rerun" -ForegroundColor DarkGray
    exit 0
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
        "--anchors", $trainCsv,
        "--val-anchor", $HoldoutAnchor,
        "--exclude-val-from-train",
        "--drop-xy",
        "--deploy-freq", "1"
    )
    if ($NoInit) {
        $trainArgs += "--no-init"
    } else {
        $trainArgs += @("--init", $InitCkpt, "--init-mode", "full")
    }
    if (-not $NoFreeze -and $Leg -eq "WG_clotrich_nplus_v2") {
        # Belt-and-suspenders with leg config_kwargs.freeze_backbone.
        $trainArgs += "--freeze-backbone"
    }

    $null = Invoke-PythonRcCheck -Label "$Leg train" -PyArgs $trainArgs

    if (-not (Test-Path $ArmCkpt)) {
        Write-Host "[ERR] $Leg failed to produce checkpoint" -ForegroundColor Red
        exit 1
    }
} elseif (-not (Test-Path $ArmCkpt)) {
    Write-Host "[ERR] $Leg missing ckpt for -EvalOnly" -ForegroundColor Red
    exit 1
}

$null = Invoke-PythonRcCheck -Label "$Leg cold eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $ArmCkpt,
    "--no-baseline",
    "--anchors", $HoldoutAnchor,
    "--out", $ArmHold
)

if (-not $SkipViz) {
    $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/mat_growth"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    $vizOut = Join-Path $vizDir "clot_ladder_${Leg}_$HoldoutAnchor.png"
    Write-Host "[NEW] ladder viz -> $vizOut" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", $Leg,
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] eval=$ArmHold" -ForegroundColor DarkGray

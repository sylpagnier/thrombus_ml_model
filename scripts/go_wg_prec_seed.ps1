param(
    [int]    $Epochs         = 25,
    [int]    $EarlyStop      = 8,
    [double] $Lr             = 1e-4,
    [string] $TrainAnchors   = "patient005,patient006,patient010",
    [string] $HoldoutAnchor  = "patient020",
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_prec_seed",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth",
    [string] $Leg            = "WG_prec_seed",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit
)

# Train-time sparse commitment on the prec-iter stack (seed-then-frontier).
# Post-hoc eval masking stalled the front; this teaches the gate under the same knobs.
# Primary: WG_prec_seed (fh=1, tk=0.05). A/B: -Leg WG_prec_seed_fh2 | WG_prec_seed_tk02
#   .\scripts\go_wg_prec_seed.ps1 -Epochs 25 -EarlyStop 8 -Fresh

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$allowed = @("WG_prec_seed", "WG_prec_seed_fh2", "WG_prec_seed_tk02")
if ($allowed -notcontains $Leg) {
    Write-Host "[ERR] Leg must be one of: $($allowed -join ', ')" -ForegroundColor Red
    exit 1
}

$OutDir = Join-Path $RepoRoot $RunRoot
$ArmDir = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
$ArmLog = Join-Path $ArmDir "train_log.jsonl"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

$trainList = @($TrainAnchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$forbidden = @("patient002", "patient023", $HoldoutAnchor)
foreach ($bad in $forbidden) {
    if ($trainList -contains $bad) {
        Write-Host "[ERR] train list must not include $bad" -ForegroundColor Red
        exit 1
    }
}
if ($trainList.Count -lt 2 -or $trainList.Count -gt 8) {
    Write-Host "[ERR] prec-seed expects 2-8 train vessels, got $($trainList.Count)" -ForegroundColor Red
    exit 1
}
$trainCsv = [string]::Join(",", $trainList)

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing prec_iter warm-start ckpt: $InitCkpt" -ForegroundColor Red
    exit 1
}

$featfixTrainLeak = @("patient005", "patient006", "patient010", "patient023", "patient002")
if (-not $NoInit -and ($featfixTrainLeak -contains $HoldoutAnchor)) {
    Write-Host "[ERR] holdout=$HoldoutAnchor was in featfix_03 train set; warm-start would leak." -ForegroundColor Red
    exit 1
}

Write-Host "[NEW] wg_prec_seed ($Leg): $Epochs ep / ES $EarlyStop / lr=$Lr" -ForegroundColor Cyan
Write-Host "[i] goal=train seed-then-frontier under typed frontier_hops + nucleation_topk" -ForegroundColor DarkGray
Write-Host "[i] stack=WG_prec_iter + sparse commit ON (no neighbor_commit_gate; warm-start safe)" -ForegroundColor DarkGray
Write-Host "[i] train($($trainList.Count))=$trainCsv" -ForegroundColor DarkGray
Write-Host "[i] val/holdout=$HoldoutAnchor (primary gate)" -ForegroundColor DarkGray
Write-Host "[i] success: cold patient020 clot score up vs post-hoc underseed; watch mat_seed + front_spd" -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random" -ForegroundColor DarkGray
} else {
    Write-Host "[i] init=$InitCkpt" -ForegroundColor DarkGray
}

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host "[skip] $Leg already completed; pass -Fresh to rerun" -ForegroundColor DarkGray
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

    $null = Invoke-PythonRcCheck -Label "$Leg train" -PyArgs $trainArgs

    if (-not (Test-Path $ArmCkpt)) {
        Write-Host "[ERR] $Leg failed to produce checkpoint (all epochs mass-rejected?)" -ForegroundColor Red
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
    $vizOut = Join-Path $vizDir "clot_ladder_prec_seed_$HoldoutAnchor.png"
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", "prec_seed",
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] eval=$ArmHold" -ForegroundColor DarkGray

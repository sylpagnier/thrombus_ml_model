param(
    [int]    $Epochs         = 25,
    [int]    $EarlyStop      = 8,
    [double] $Lr             = 1e-4,
    [string] $TrainAnchors   = "patient005,patient006,patient010",
    [string] $HoldoutAnchor  = "patient020",
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_prec_pocket",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth",
    [string] $Leg            = "WG_prec_pocket",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit
)

# Multi-pocket exclusive contrast from WG_prec_iter floor (NOT physfp/seed/front).
# Soft-penalize Mat outside k-hop of GT first-seed; no hard frontier mask.
# Gate: primary deploy_clot_f1 on patient020; mass hard [0.5,1.5]; FN hard max 80.
#   .\scripts\go_wg_prec_pocket.ps1 -Fresh

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Leg -ne "WG_prec_pocket") {
    Write-Host ("[ERR] This launcher is for WG_prec_pocket only (got {0})" -f $Leg) -ForegroundColor Red
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
        Write-Host ("[ERR] train list must not include {0}" -f $bad) -ForegroundColor Red
        exit 1
    }
}
if ($trainList.Count -lt 2 -or $trainList.Count -gt 8) {
    Write-Host ("[ERR] pocket expects 2-8 train vessels, got {0}" -f $trainList.Count) -ForegroundColor Red
    exit 1
}
$trainCsv = [string]::Join(",", $trainList)

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host ("[ERR] Missing prec_iter warm-start ckpt: {0}" -f $InitCkpt) -ForegroundColor Red
    exit 1
}

Write-Host ("[NEW] {0}: {1} ep / ES {2} / lr={3}" -f $Leg, $Epochs, $EarlyStop, $Lr) -ForegroundColor Cyan
Write-Host "[i] goal=multi-pocket selection via exclusive soft contrast (not physfp/seed/front)" -ForegroundColor DarkGray
Write-Host "[i] stack=WG_prec_iter + pocket_contrast_weight=0.35 hops=4; fh=0 tk=0 seed_aux=0 physfp=0" -ForegroundColor DarkGray
Write-Host "[i] gate=primary deploy_clot_f1; mass hard [0.5,1.5]; FN hard max 80" -ForegroundColor DarkGray
Write-Host ("[i] train({0})={1} holdout={2}" -f $trainList.Count, $trainCsv, $HoldoutAnchor) -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random" -ForegroundColor DarkGray
} else {
    Write-Host ("[i] init={0}" -f $InitCkpt) -ForegroundColor DarkGray
}

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host ("[skip] {0} already completed; pass -Fresh to rerun" -f $Leg) -ForegroundColor DarkGray
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
        Write-Host ("[ERR] {0} failed to produce checkpoint (all epochs mass/FN-rejected?)" -f $Leg) -ForegroundColor Red
        exit 1
    }
} elseif (-not (Test-Path $ArmCkpt)) {
    Write-Host ("[ERR] {0} missing ckpt for -EvalOnly" -f $Leg) -ForegroundColor Red
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
    $vizOut = Join-Path $vizDir ("clot_ladder_prec_pocket_{0}.png" -f $HoldoutAnchor)
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", "prec_pocket",
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host ("[OK] {0} complete" -f $Leg) -ForegroundColor Green
Write-Host ("[i] ckpt={0}" -f $ArmCkpt) -ForegroundColor DarkGray
Write-Host ("[i] eval={0}" -f $ArmHold) -ForegroundColor DarkGray

param(
    [string] $Leg         = "WG_sched_sample",
    [int]    $Epochs      = 30,
    [int]    $EarlyStop   = 15,
    [int]    $MaxWindows  = 24,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020,patient034",
    [string] $RunRoot    = "outputs/biochem/eda/wall_gen",
    [switch] $Fresh,
    [switch] $EvalOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot $Leg
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$arm  = "cold"
$Ckpt = Join-Path $OutDir "best.pth"

Write-Host "[i] Wall-gen probe: leg=$Leg, arm=$arm" -ForegroundColor Cyan
Write-Host "[i] train   = $TrainAnchors" -ForegroundColor DarkGray
Write-Host "[i] holdout = $HoldoutAnchors" -ForegroundColor DarkGray
Write-Host "[i] cold init (--no-init)" -ForegroundColor DarkGray

if (-not $EvalOnly) {
    if ($Fresh) { Remove-Item -Force $Ckpt -ErrorAction SilentlyContinue }
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", $Leg,
        "--anchors", $TrainAnchors,
        "--val-anchor", $ValAnchor,
        "--exclude-val-from-train",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--out", $Ckpt,
        "--no-init"
    )
    $null = Invoke-PythonRcCheck -Label "wall_gen probe $arm train" -PyArgs $trainArgs
}

if (-not (Test-Path $Ckpt)) { throw "wall_gen ckpt missing: $Ckpt" }

$holdJson = Join-Path $OutDir "eval_holdout_$arm.json"
$null = Invoke-PythonRcCheck -Label "wall_gen $arm holdout eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $Ckpt,
    "--mat-leg", $Leg,
    "--no-baseline",
    "--anchors", $HoldoutAnchors,
    "--out", $holdJson
)

Write-Host "[OK] wall_gen done -> $OutDir" -ForegroundColor Green
Write-Host "[i] holdout: $holdJson" -ForegroundColor DarkGray

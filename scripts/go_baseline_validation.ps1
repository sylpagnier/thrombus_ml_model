param(
    [int]    $Epochs       = 10,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020",
    [string] $RunRoot      = "outputs/biochem/eda/baseline_validation"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$arms = @(
    "01_wcv7_fresh",
    "02_wcv7_dropxy",
    "03_wcv7_dropxy_kine",
    "04_wcv7_dropxy_kine_wallonly"
)

$started = Get-Date
Write-Host "[NEW] baseline_validation: $($arms.Count) arms, $Epochs ep" -ForegroundColor Cyan

foreach ($armId in $arms) {
    $armDir = Join-Path $OutDir "VAL_baseline_$armId"
    $armCkpt = Join-Path $armDir "best.pth"
    $armHold = Join-Path $armDir "eval_holdout_cold.json"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null
    
    if (Test-Path $armHold) {
        Write-Host "[i] Arm $armId already completed (eval JSON exists). Skipping!" -ForegroundColor DarkGray
        continue
    }

    Write-Host "" -ForegroundColor Cyan
    Write-Host "====== Arm ${armId} ======" -ForegroundColor Cyan
    
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", "VAL_baseline_$armId",
        "--out", $armCkpt,
        "--epochs", "$Epochs",
        "--anchors", $TrainAnchors,
        "--val-anchor", $ValAnchor,
        "--exclude-val-from-train",
        "--no-init"
    )
    
    $null = Invoke-PythonRcCheck -Label "Arm $armId train" -PyArgs $trainArgs
    
    if (-not (Test-Path $armCkpt)) { 
        Write-Host "[WARN] Arm $armId failed to produce checkpoint!" -ForegroundColor Yellow
        continue 
    }

    # Evaluate
    $null = Invoke-PythonRcCheck -Label "Arm $armId eval" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $armCkpt,
        "--no-baseline",
        "--anchors", $HoldoutAnchors,
        "--out", $armHold
    )
}

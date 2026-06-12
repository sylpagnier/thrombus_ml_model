# Phase 1 species snapshot GNN: static FI/Mat @ one macro time on wall-band subgraph.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_snapshot_s1.ps1" -Fresh
#   powershell ... -Fresh -Anchor patient007 -TimeS 5000 -Epochs 80

param(
    [string] $Anchor = "patient007",
    [double] $TimeS = 5000,
    [int] $Epochs = 80,
    [int] $WallHops = 2,
    [string] $Loss = "focal",
    [string] $Ckpt = "outputs/biochem/species_snapshot_s1/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $VizOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:SPECIES_SNAPSHOT_TIME_S = "$TimeS"
$env:SPECIES_SNAPSHOT_WALL_HOPS = "$WallHops"
$env:SPECIES_SNAPSHOT_LOSS = $Loss
$env:SPECIES_SNAPSHOT_CKPT = $Ckpt
# Mat-tuned defaults (sweep winner fi90_mat70 + mat_thresh 0.55)
$env:SPECIES_SNAPSHOT_FOCAL_ALPHA_FI = if ($env:SPECIES_SNAPSHOT_FOCAL_ALPHA_FI) { $env:SPECIES_SNAPSHOT_FOCAL_ALPHA_FI } else { "0.90" }
$env:SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT = if ($env:SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT) { $env:SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT } else { "0.70" }
$env:SPECIES_SNAPSHOT_MAT_THRESH = if ($env:SPECIES_SNAPSHOT_MAT_THRESH) { $env:SPECIES_SNAPSHOT_MAT_THRESH } else { "0.55" }

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain -and -not $VizOnly) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train species snapshot s1 anchor=$Anchor time_s=$TimeS" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_species_snapshot_gnn",
            "--anchor", $Anchor,
            "--time-s", "$TimeS",
            "--wall-hops", "$WallHops",
            "--epochs", "$Epochs",
            "--loss", $Loss,
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "species snapshot s1 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Viz species snapshot s1 ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_species_snapshot_gnn.py",
    "--anchor", $Anchor,
    "--time-s", "$TimeS",
    "--ckpt", $Ckpt
)
Invoke-PythonRcCheck -Label "species snapshot s1 viz" -PyArgs $vizArgs

Write-Host "[OK] ckpt=$Ckpt" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/species_gnn/s1_${Anchor}.png" -ForegroundColor Green

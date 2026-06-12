# Rung 4 step s5: narrow 2-ch FI/Mat band GNN on frozen s0 gate + eval + viz.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_rung4_s5.ps1" -Fresh
#   powershell ... -Fresh -Epochs 40 -ValAnchor patient007 -Recipe s5_gnn_fimat
#
# Recipe variants (T0_R4_S5_RECIPE):
#   s5_gnode_fimat  -- default: GNN delta + species/commit loss (ladder step)
#   s5_gnn_fimat    -- sweep variant (lighter w_species)
#   s5_mlp_fimat    -- MLP delta on hotspots
#   s5_gru_fimat    -- MLP delta + GRU temporal smooth

param(
    [string] $Anchor = "patient007",
    [string] $Times = "0,7,15,22,27,40,53",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 40,
    [string] $Recipe = "s5_gnode_fimat",
    [string] $Ckpt = "outputs/biochem/t0_r4_s5_gnode_fimat/best.pth",
    [switch] $SkipTrain,
    [switch] $Fresh,
    [string] $TeacherCkpt = "outputs/biochem/biochem_teacher_last.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:T0_RUNG4_STEP = "s5_gnode_fimat"
$env:T0_R4_S5_CKPT = $Ckpt
$env:T0_R4_S5_RECIPE = $Recipe

$ckptPath = Join-Path $RepoRoot $Ckpt
if ($Fresh -and (Test-Path $ckptPath)) {
    Remove-Item $ckptPath -Force
    $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
    if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
    $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
    if (Test-Path $logPath) { Remove-Item $logPath -Force }
}

if (-not $SkipTrain) {
    if (-not (Test-Path $ckptPath)) {
        Write-Host "[NEW] Train s5_gnode_fimat recipe=$Recipe (val=$ValAnchor, epochs=$Epochs)" -ForegroundColor Cyan
        $trainArgs = @(
            "-m", "src.training.train_t0_r4_sweep_leg",
            "--recipe", $Recipe,
            "--val-anchor", $ValAnchor,
            "--epochs", "$Epochs",
            "--early-stop", "12",
            "--out", $Ckpt
        )
        Invoke-PythonRcCheck -Label "rung4 s5 train" -PyArgs $trainArgs
    } else {
        Write-Host "[skip] checkpoint exists: $Ckpt (use -Fresh to retrain)" -ForegroundColor Yellow
    }
}

Write-Host "[NEW] Rung 4.s5 eval ($Anchor)" -ForegroundColor Cyan
$evalArgs = @(
    "scripts/eval_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--times", $Times,
    "--step", "s5_gnode_fimat"
)
if ($TeacherCkpt) { $evalArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s5 eval" -PyArgs $evalArgs

Write-Host "[NEW] Rung 4.s5 viz ($Anchor)" -ForegroundColor Cyan
$vizArgs = @(
    "scripts/viz_t0_rung4_step.py",
    "--anchor", $Anchor,
    "--max-frames", "10",
    "--step", "s5_gnode_fimat"
)
if ($TeacherCkpt) { $vizArgs += @("--teacher-ckpt", $TeacherCkpt) }
Invoke-PythonRcCheck -Label "rung4 s5 viz" -PyArgs $vizArgs

Write-Host "[OK] ckpt=$Ckpt recipe=$Recipe" -ForegroundColor Green
Write-Host "[OK] eval=outputs/biochem/clot_trigger/t0_rung4_s5_gnode_fimat_${Anchor}.json" -ForegroundColor Green
Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/t0_rung4_s5_gnode_fimat_${Anchor}.png" -ForegroundColor Green

# K4 -> K5 staged split-head experiment (wall first, then clot + physics).
# Delete outputs/biochem/*.pth (keep kinematics_best.pth) before a full fresh ladder.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4k5_split_head_staged.ps1"
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_k4k5_split_head_staged.ps1" -K4Epochs 18 -K5Epochs 15

param(
    [int] $K4Epochs = 12,
    [int] $K5Epochs = 15
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

& "$here\go_k4_wall_head_only.ps1" -Epochs $K4Epochs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Remove-Item Env:BIOCHEM_MU_TRAIN_WALL_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_MU_TRAIN_CLOT_ONLY -ErrorAction SilentlyContinue

& "$here\go_k5_clot_head_physics.ps1" -Epochs $K5Epochs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "K4+K5 staged run complete." -ForegroundColor Green

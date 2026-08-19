param (
    [int]$Epochs = 25,
    [string]$Flow = "pred",
    [string]$ModelType = "global"
)

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " TUNING DIFFERENTIABLE WALL MODEL" -ForegroundColor Cyan
Write-Host " Epochs: $Epochs | Flow: $Flow | Model Type: $ModelType" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

python scripts/tune_differentiable_wall_model.py --epochs $Epochs --flow $Flow --model-type $ModelType

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Tuning completed successfully." -ForegroundColor Green
} else {
    Write-Host "[WARN] Tuning exited with code $LASTEXITCODE." -ForegroundColor Yellow
}

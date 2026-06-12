# M0 wall-band species: train + eval + viz (fimat 2-ch vs cascade4 4-ch).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_wall_band_species_m0.ps1" -ChannelSet fimat -Fresh
#   powershell ... -ChannelSet cascade4 -Fresh
#   powershell ... -CompareBoth -Fresh    # train/eval both channel sets

param(
    [ValidateSet("fimat", "cascade4", "both")]
    [string] $ChannelSet = "fimat",
    [string] $Anchor = "patient007",
    [string] $ValAnchor = "patient007",
    [int] $Epochs = 30,
    [switch] $SkipTrain,
    [switch] $Fresh,
    [switch] $CompareBoth
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }

function Invoke-M0Leg {
    param(
        [string] $SetId
    )
    $ckptRel = "outputs/biochem/wall_band_species_m0_$SetId/best.pth"
    $ckptPath = Join-Path $RepoRoot $ckptRel
    $env:WALL_BAND_M0_CKPT = $ckptRel
    $env:WALL_BAND_M0_CHANNEL_SET = $SetId

    if ($Fresh -and (Test-Path $ckptPath)) {
        Remove-Item $ckptPath -Force
        $jsonSide = Join-Path (Split-Path $ckptPath) "best.json"
        if (Test-Path $jsonSide) { Remove-Item $jsonSide -Force }
        $logPath = Join-Path (Split-Path $ckptPath) "train_log.jsonl"
        if (Test-Path $logPath) { Remove-Item $logPath -Force }
    }

    if (-not $SkipTrain) {
        if (-not (Test-Path $ckptPath)) {
            Write-Host "[NEW] Train M0 channel_set=$SetId (val=$ValAnchor)" -ForegroundColor Cyan
            Invoke-PythonRcCheck -Label "M0 train $SetId" -PyArgs @(
                "-m", "src.training.train_wall_band_species_m0",
                "--channel-set", $SetId,
                "--val-anchor", $ValAnchor,
                "--epochs", "$Epochs",
                "--out", $ckptRel
            )
        } else {
            Write-Host "[skip] train $SetId ckpt exists (use -Fresh)" -ForegroundColor Yellow
        }
    }

    Write-Host "[NEW] M0 eval $SetId ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "M0 eval $SetId" -PyArgs @(
        "scripts/eval_wall_band_species_m0.py",
        "--anchor", $Anchor,
        "--ckpt", $ckptRel
    )

    Write-Host "[NEW] M0 viz $SetId ($Anchor)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label "M0 viz $SetId" -PyArgs @(
        "scripts/viz_wall_band_species_m0.py",
        "--anchor", $Anchor,
        "--ckpt", $ckptRel,
        "--max-frames", "10"
    )

    Write-Host "[OK] $SetId ckpt=$ckptRel" -ForegroundColor Green
    Write-Host "[OK] eval=outputs/biochem/clot_trigger/wall_band_m0_${SetId}_${Anchor}.json" -ForegroundColor Green
    Write-Host "[OK] viz=outputs/biochem/viz/clot_trigger/wall_band_m0_${SetId}_${Anchor}.png" -ForegroundColor Green
}

$sets = @()
if ($CompareBoth -or $ChannelSet -eq "both") {
    $sets = @("fimat", "cascade4")
} else {
    $sets = @($ChannelSet)
}

foreach ($s in $sets) {
    Invoke-M0Leg -SetId $s
}

if ($sets.Count -gt 1) {
    Write-Host "[i] Compare eval JSONs for fimat vs cascade4 on $Anchor" -ForegroundColor Cyan
}

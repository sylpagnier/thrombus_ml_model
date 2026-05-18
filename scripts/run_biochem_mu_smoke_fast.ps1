# Fast μ smoke test (minutes) for train_biochem_corrector.
# Purpose: verify we can move μ-directed training signal quickly.
# Not a generalization benchmark.
#
# Usage (repo root):
#   .\scripts\run_biochem_mu_smoke_fast.ps1
#   .\scripts\run_biochem_mu_smoke_fast.ps1 -LossIsolate MU_SI -TeacherEpochs 4
#   .\scripts\run_biochem_mu_smoke_fast.ps1 -UseDeltaMuHead

param(
    [ValidateSet("MU_LOG", "MU_SI")]
    [string] $LossIsolate = "MU_LOG",
    [int] $TeacherEpochs = 3,
    [switch] $UseDeltaMuHead,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "μ smoke fast: signal sanity only (not generalization)." -ForegroundColor Yellow
Write-Host "Target: finite run + directional movement in L_Back / μ isolate loss in 2-4 epochs." -ForegroundColor Yellow

Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$warmStart = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$useWarm = Test-Path $warmStart

$env:BIOCHEM_RUN_NOTE = "mu_smoke_fast_$LossIsolate"
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_PRESET = ""
$env:BIOCHEM_COMPLEXITY_STEP = "2"
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_LOSS_ISOLATE = $LossIsolate
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
$env:BIOCHEM_TEACHER_SKIP_VAL = "1"
$env:BIOCHEM_TEACHER_VAL_EVERY = "99"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_MAX_LOAD_VESSELS = "1"
$env:BIOCHEM_MAX_LOAD_SHUFFLE = "0"
$env:BIOCHEM_LOW_ANCHOR_MODE = "1"
$env:BIOCHEM_DATALOADER_WORKERS = "0"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "4"
$env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
$env:BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
$env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
$env:BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
$env:BIOCHEM_DETACH_MACRO_STATE = "1"
$env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "12"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "2"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_MU_SI_MULTI_STEP = "1"
$env:BIOCHEM_MU_SI_HUBER_DELTA = "0.25"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = if ($LossIsolate -eq "MU_SI") { "8.0" } else { "0.0" }
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = if ($LossIsolate -eq "MU_LOG") { "2.0" } else { "0.0" }
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_MU_PATH_LR_MULT = "1.0"
$env:BIOCHEM_DEBUG = "0"

if ($UseDeltaMuHead) {
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP = "1.5"
} else {
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "0"
}

if ($useWarm) {
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    Write-Host "Warm-start enabled: $warmStart" -ForegroundColor Cyan
} else {
    $env:BIOCHEM_SKIP_PRETRAIN = "0"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    Write-Host "Warm-start not found; pretrain will run (slower)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Running: python -m src.training.train_biochem_corrector --new" -ForegroundColor Cyan
Write-Host "Loss isolate=$LossIsolate | epochs=$TeacherEpochs | delta_mu_head=$($env:BIOCHEM_USE_DELTA_MU_HEAD)"
Write-Host ""

python -m src.training.train_biochem_corrector --new @ExtraArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

function Get-LatestBiochemMetricsPath {
    param([string] $ReportsRoot)
    $timestampDir = Get-ChildItem -Path $ReportsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{8}T\d{6}Z$' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $timestampDir) {
        $p = Join-Path $timestampDir.FullName "metrics.jsonl"
        if (Test-Path $p) { return $p }
    }
    $fallback = Join-Path $ReportsRoot "metrics.jsonl"
    if (Test-Path $fallback) { return $fallback }
    return $null
}

function Read-Jsonl {
    param([string] $Path)
    $rows = @()
    Get-Content -Path $Path -ErrorAction SilentlyContinue | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) { return }
        try {
            $rows += ($line | ConvertFrom-Json)
        } catch {
            # ignore malformed lines
        }
    }
    return $rows
}

Write-Host ""
$reportsRoot = Join-Path $RepoRoot "outputs\reports\training\biochem"
$metricsPath = Get-LatestBiochemMetricsPath -ReportsRoot $reportsRoot
if ($null -eq $metricsPath) {
    Write-Host "Done. No metrics.jsonl found under $reportsRoot" -ForegroundColor Yellow
    exit 0
}

$rows = Read-Jsonl -Path $metricsPath
$teacherRows = @($rows | Where-Object { $_.stage -eq "teacher" })
if ($teacherRows.Count -eq 0) {
    Write-Host "Done. Metrics found but no teacher rows: $metricsPath" -ForegroundColor Yellow
    exit 0
}

$showRows = @($teacherRows | Sort-Object epoch | Select-Object -Last ([Math]::Min(6, $teacherRows.Count)))
Write-Host "Done. Auto-summary from: $metricsPath" -ForegroundColor Green
Write-Host "ep | L_back | val_logMAE | wall | high | mu1 | mu2 | learned | flow_imb"
foreach ($r in $showRows) {
    $ep = $r.epoch
    $lb = if ($null -ne $r.train_L_back_avg) { "{0:E3}" -f [double]$r.train_L_back_avg } else { "nan" }
    $va = if ($null -ne $r.val_mu_log_mae) { "{0:F4}" -f [double]$r.val_mu_log_mae } else { "n/a" }
    $vw = if ($null -ne $r.val_mu_log_mae_wall) { "{0:F4}" -f [double]$r.val_mu_log_mae_wall } else { "n/a" }
    $vh = if ($null -ne $r.val_mu_log_mae_high_mu) { "{0:F4}" -f [double]$r.val_mu_log_mae_high_mu } else { "n/a" }
    $m1 = if ($null -ne $r.dbg_mu1_mean) { "{0:E2}" -f [double]$r.dbg_mu1_mean } else { "n/a" }
    $m2 = if ($null -ne $r.dbg_mu2_mean) { "{0:E2}" -f [double]$r.dbg_mu2_mean } else { "n/a" }
    $ml = if ($null -ne $r.dbg_mu_learned_mean) { "{0:E2}" -f [double]$r.dbg_mu_learned_mean } else { "n/a" }
    $fi = if ($null -ne $r.dbg_flux_imbalance_mean) { "{0:E2}" -f [double]$r.dbg_flux_imbalance_mean } else { "n/a" }
    Write-Host "$ep | $lb | $va | $vw | $vh | $m1 | $m2 | $ml | $fi"
}

Write-Host "Smoke pass: finite run + downward L_back and/or val_logMAE, with non-trivial mu components and bounded flow imbalance."

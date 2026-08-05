param(
    [int]    $Epochs         = 30,
    [int]    $EarlyStop      = 15,
    [string] $TrainAnchors   = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor      = "patient020",
    [string] $HoldoutAnchors = "patient020",
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_sweep_v3",
    [string] $InitCkpt       = "outputs/biochem/biochem_gnn/wall_gen_baseline/species/best.pth",
    [string] $ArmFilter      = "",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $NoInit
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing wall-gen baseline ckpt: $InitCkpt" -ForegroundColor Red
    Write-Host "[i] Promote first: python scripts/promote_wall_gen_baseline.py --eval-json <p020 eval>" -ForegroundColor DarkGray
    exit 1
}

Write-Host "[NEW] phase1_sweep_v3: $Epochs ep / ES $EarlyStop" -ForegroundColor Cyan
Write-Host "[i] base=FS_ab_coupled wall_gen (auto+coupled, drop-xy)" -ForegroundColor DarkGray
Write-Host "[i] train=$TrainAnchors val=$ValAnchor holdout=$HoldoutAnchors" -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random (--NoInit)" -ForegroundColor DarkGray
} else {
    Write-Host "[i] init=$InitCkpt" -ForegroundColor DarkGray
}

# Prefix-only listing avoids WC_v3_* off-wall legs matching substring "v3".
$armsStr = (python -m src.training.train_species_pushforward_continuous --list-legs WG_sweep_v3) | Out-String
$armKeys = @($armsStr.Trim() -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("[") -and $_.StartsWith("WG_sweep_v3_") })

if ($ArmFilter) {
    $filters = @($ArmFilter.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $armKeys = @($armKeys | Where-Object {
        $filters -contains $_ -or $filters -contains $_.Replace("WG_sweep_v3_", "")
    })
}

if ($armKeys.Count -lt 1) {
    Write-Host "[ERR] No WG_sweep_v3_* arms selected." -ForegroundColor Red
    exit 1
}

Write-Host "[i] arms ($($armKeys.Count)): $($armKeys -join ', ')" -ForegroundColor Cyan

foreach ($armId in $armKeys) {
    $armDir = Join-Path $OutDir $armId
    $armCkpt = Join-Path $armDir "best.pth"
    $armHold = Join-Path $armDir "eval_holdout_cold.json"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null

    if ($Fresh) {
        Remove-Item -Force $armCkpt, $armHold -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $armDir "best.json"), (Join-Path $armDir "train_log.jsonl") -ErrorAction SilentlyContinue
    }

    if ((Test-Path $armHold) -and -not $Fresh -and -not $EvalOnly) {
        Write-Host "[skip] $armId already completed (eval JSON exists)" -ForegroundColor DarkGray
        continue
    }

    Write-Host ""
    Write-Host "====== Arm ${armId} ======" -ForegroundColor Cyan

    if (-not $EvalOnly) {
        $trainArgs = @(
            "-m", "src.training.train_species_pushforward_continuous",
            "--phase", "biochem_gnn",
            "--recipe", "mat_growth_simple",
            "--leg", $armId,
            "--out", $armCkpt,
            "--epochs", "$Epochs",
            "--early-stop", "$EarlyStop",
            "--anchors", $TrainAnchors,
            "--val-anchor", $ValAnchor,
            "--exclude-val-from-train",
            "--drop-xy",
            "--deploy-freq", "1"
        )
        if ($NoInit) {
            $trainArgs += "--no-init"
        } else {
            $trainArgs += @("--init", $InitCkpt, "--init-mode", "full")
        }

        $null = Invoke-PythonRcCheck -Label "Arm $armId train" -PyArgs $trainArgs

        if (-not (Test-Path $armCkpt)) {
            Write-Host "[WARN] Arm $armId failed to produce checkpoint" -ForegroundColor Yellow
            continue
        }
    } elseif (-not (Test-Path $armCkpt)) {
        Write-Host "[WARN] Arm $armId missing ckpt for -EvalOnly" -ForegroundColor Yellow
        continue
    }

    # Deploy-faithful cold eval: do NOT pass --mat-leg (avoids re-applying train knobs).
    # _apply_ckpt_recipe forces auto flow + corrector coupling.
    $null = Invoke-PythonRcCheck -Label "Arm $armId eval" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $armCkpt,
        "--no-baseline",
        "--anchors", $HoldoutAnchors,
        "--out", $armHold
    )
}

Write-Host ""
Write-Host "[NEW] summarizing phase1_sweep_v3" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "aggregate_sweep_v3" -PyArgs @(
    "scripts/aggregate_sweep_v3.py"
)

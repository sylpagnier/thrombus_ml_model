# Compare passive teacher species quality with/without L_Data_Bio backward.
# Runs per leg:
#   1) teacher-only passive leg (stronger quick recipe)
#   2) dump teacher species to anchors
#   3) quick clot-phi train on dumped anchors
#   4) multi-anchor eval
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_species_focus_compare.ps1"

param(
    [string[]] $Legs = @("lbio_on", "lbio_off"),
    [int] $TeacherEpochs = 10,
    [int] $ClotEpochs = 20,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutRoot = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Run-TeacherLeg {
    param(
        [string] $LegName,
        [string] $DumpDir,
        [double] $BioWeight
    )
    Write-Host "[NEW] passive teacher leg=$LegName teacher_ep=$TeacherEpochs bio_w=$BioWeight" -ForegroundColor Cyan

    # Base passive config (non-interactive)
    $env:KINEMATICS_USE_HARD_BCS = "1"
    $env:KINEMATICS_USE_WIDTH_PRIORS = "1"
    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_RUN_NOTE = "passive_species_focus_$LegName"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_TEACHER_LR = "3e-4"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.25"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "$BioWeight"

    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name "passive_species_focus_$LegName"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    python scripts/dump_teacher_species_to_anchors.py `
        --teacher outputs/biochem/biochem_teacher_last.pth `
        --out-dir $DumpDir `
        --device cuda --time-stride $DumpStride --min-steps $DumpMinSteps --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Run-ClotLeg {
    param(
        [string] $LegName,
        [string] $DumpDir,
        [string] $LegOutDir
    )
    Write-Host "[NEW] clot-phi leg=$LegName clot_ep=$ClotEpochs" -ForegroundColor Cyan

    Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
        Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
    }
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_ANCHOR_DIR = $DumpDir
    $env:CLOT_PHI_MODEL = "mlp"
    $env:CLOT_PHI_HIDDEN = "32"
    $env:CLOT_PHI_MLP_DEPTH = "2"
    $env:CLOT_PHI_DROPOUT = "0.15"
    $env:CLOT_PHI_LR = "1e-3"
    $env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
    $env:CLOT_PHI_EPOCHS = "$ClotEpochs"
    $env:CLOT_PHI_SPECIES_FEATURES = "0"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_ANCHOR_BALANCED = "1"
    $env:CLOT_PHI_BIO_FI_WEIGHT = "2.0"
    $env:CLOT_PHI_BIO_MAT_WEIGHT = "2.0"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/passive_species_focus_compare"
    $env:CLOT_PHI_SWEEP_LEG = $LegName
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"

    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $ckpt = Join-Path $LegOutDir "clot_phi_best.pth"
    $evalOut = Join-Path $LegOutDir "multi_anchor.jsonl"
    python scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Summarize-Leg {
    param([string] $LegName, [string] $LegOutDir)
    $jsonl = Join-Path $LegOutDir "multi_anchor.jsonl"
    if (-not (Test-Path $jsonl)) { return $null }
    $rows = Get-Content $jsonl | ForEach-Object { $_ | ConvertFrom-Json }
    if (-not $rows) { return $null }
    $f1 = ($rows | ForEach-Object { [double]$_.val.clot_f1 } | Measure-Object -Average).Average
    $minf1 = ($rows | ForEach-Object { [double]$_.val.clot_f1 } | Measure-Object -Minimum).Minimum
    $mae = ($rows | ForEach-Object { [double]$_.val.mu_log_mae } | Measure-Object -Average).Average
    $score = ($rows | ForEach-Object { [double]$_.val_score } | Measure-Object -Average).Average
    return [pscustomobject]@{
        leg = $LegName
        mean_f1 = [math]::Round($f1, 3)
        min_f1 = [math]::Round($minf1, 3)
        mean_logMAE = [math]::Round($mae, 3)
        mean_score = [math]::Round($score, 3)
    }
}

$summary = @()
foreach ($leg in $Legs) {
    $legOutDir = Join-Path $OutRoot $leg
    New-Item -ItemType Directory -Force -Path $legOutDir | Out-Null
    $dumpDir = Join-Path $OutRoot ("anchors_" + $leg)

    switch ($leg) {
        "lbio_on" {
            Run-TeacherLeg -LegName $leg -DumpDir $dumpDir -BioWeight 1.0
        }
        "lbio_off" {
            Run-TeacherLeg -LegName $leg -DumpDir $dumpDir -BioWeight 0.0
        }
        default {
            Write-Host "[WARN] unknown leg=$leg, skipping" -ForegroundColor Yellow
            continue
        }
    }

    Run-ClotLeg -LegName $leg -DumpDir $dumpDir -LegOutDir $legOutDir
    $s = Summarize-Leg -LegName $leg -LegOutDir $legOutDir
    if ($s -ne $null) {
        $summary += $s
        Write-Host "[i]  $($s.leg): mean_f1=$($s.mean_f1) min_f1=$($s.min_f1) mean_logMAE=$($s.mean_logMAE) mean_score=$($s.mean_score)" -ForegroundColor Yellow
    }
}

if ($summary.Count -gt 0) {
    $summary | Format-Table -AutoSize
    $summary | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $OutRoot "summary.json") -Encoding utf8
    Write-Host "[OK]  wrote $(Join-Path $OutRoot "summary.json")" -ForegroundColor Green
}

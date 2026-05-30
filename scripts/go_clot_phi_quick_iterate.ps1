# Quick iteration ladder before long overnight runs.
#   powershell -File .\scripts\go_clot_phi_quick_iterate.ps1
#
# Legs:
#   oracle_gt   - clot-phi on GT anchors (species-feature ceiling)
#   passive_tf08 - passive teacher 8ep, TF>=0.2, L_Data_Bio (resume from best)
#   staged      - staged mu-then-phi on teacher-species cache (stride 36)

param(
    [string[]] $Legs = @("oracle_gt", "passive_tf08", "staged"),
    [int] $TeacherEpochs = 8,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutRoot = Join-Path $RepoRoot "outputs\biochem\quick_iterate"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Invoke-ClotTrain {
    param(
        [string] $Leg,
        [hashtable] $Env,
        [string] $CkptOut,
        [int] $Epochs
    )
    Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    foreach ($k in $Env.Keys) { Set-Item -Path "Env:$k" -Value $Env[$k] }
    $env:CLOT_PHI_EPOCHS = "$Epochs"
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth", "outputs\biochem\clot_phi_train_log.jsonl"
    Write-Host "[NEW] clot-phi leg=$Leg epochs=$Epochs" -ForegroundColor Cyan
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item -Force "outputs\biochem\clot_phi_best.pth" $CkptOut
}

function Invoke-MultiEval {
    param([string] $Ckpt, [string] $JsonOut)
    python scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $JsonOut
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Summarize-Jsonl {
    param([string] $Path)
    if (-not (Test-Path $Path)) { return $null }
    $rows = Get-Content $Path | ForEach-Object { $_ | ConvertFrom-Json }
    if (-not $rows) { return $null }
    $f1 = ($rows | ForEach-Object { [double]$_.val.clot_f1 } | Measure-Object -Average).Average
    $minf1 = ($rows | ForEach-Object { [double]$_.val.clot_f1 } | Measure-Object -Minimum).Minimum
    $mae = ($rows | ForEach-Object { [double]$_.val.mu_log_mae } | Measure-Object -Average).Average
    $sc = ($rows | ForEach-Object { [double]$_.val_score } | Measure-Object -Average).Average
  return [pscustomobject]@{ mean_f1 = [math]::Round($f1, 3); min_f1 = [math]::Round($minf1, 3); mean_logMAE = [math]::Round($mae, 3); mean_score = [math]::Round($sc, 3) }
}

function Invoke-PassiveTeacher {
    param([string] $Leg, [int] $Epochs)
    Write-Host "[NEW] passive teacher leg=$Leg epochs=$Epochs" -ForegroundColor Cyan
    $env:KINEMATICS_USE_HARD_BCS = "1"
    $env:KINEMATICS_USE_WIDTH_PRIORS = "1"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = "quick_iterate_$Leg"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "2"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    Remove-Item Env:BIOCHEM_PRESET -ErrorAction SilentlyContinue

    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "0.2"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "3"
    $env:BIOCHEM_TEACHER_LR = "5e-4"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue

    python -m src.training.train_biochem_corrector --epochs $Epochs --save-best --run-name "quick_$Leg"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$summary = @()

foreach ($leg in $Legs) {
    $legDir = Join-Path $OutRoot $leg
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    $evalJson = Join-Path $legDir "multi_anchor.jsonl"

    switch ($leg) {
        "oracle_gt" {
            $envMap = @{
                CLOT_PHI_ANCHOR_DIR = "data/processed/graphs_biochem_anchors"
                CLOT_PHI_SPECIES_FEATURES = "1"
                CLOT_PHI_JOINT_BIO = "0"
                CLOT_PHI_PHYSICS_BLEND = "1"
                CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
                CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
                CLOT_PHI_PHYSICS_GELATION_GATE = "1"
                CLOT_PHI_THRESH_SI = "0.045"
            }
            Invoke-ClotTrain -Leg $leg -Env $envMap -CkptOut $ckpt -Epochs $ClotEpochs
            Invoke-MultiEval -Ckpt $ckpt -JsonOut $evalJson
        }
        "passive_tf08" {
            Invoke-PassiveTeacher -Leg $leg -Epochs $TeacherEpochs
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "outputs\biochem\anchors_teacher_species"
            python scripts/dump_teacher_species_to_anchors.py `
                --teacher outputs/biochem/biochem_teacher_last.pth `
                --out-dir outputs/biochem/anchors_teacher_species `
                --device cuda --time-stride $DumpStride --min-steps $DumpMinSteps --force
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            $envMap = @{
                CLOT_PHI_ANCHOR_DIR = "outputs/biochem/anchors_teacher_species"
                CLOT_PHI_SPECIES_FEATURES = "0"
                CLOT_PHI_JOINT_BIO = "1"
                CLOT_PHI_ANCHOR_BALANCED = "1"
                CLOT_PHI_BIO_FI_WEIGHT = "2.0"
                CLOT_PHI_BIO_MAT_WEIGHT = "2.0"
                CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
                CLOT_PHI_PHYSICS_BLEND = "1"
                CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
                CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
                CLOT_PHI_PHYSICS_GELATION_GATE = "1"
                CLOT_PHI_THRESH_SI = "0.045"
            }
            Invoke-ClotTrain -Leg $leg -Env $envMap -CkptOut $ckpt -Epochs $ClotEpochs
            Invoke-MultiEval -Ckpt $ckpt -JsonOut $evalJson
        }
        "staged" {
            if (-not (Test-Path "outputs/biochem/anchors_teacher_species/patient001.pt")) {
                python scripts/dump_teacher_species_to_anchors.py `
                    --teacher outputs/biochem/biochem_teacher_last.pth `
                    --out-dir outputs/biochem/anchors_teacher_species `
                    --device cuda --time-stride $DumpStride --min-steps $DumpMinSteps --force
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            $stageA = Join-Path $legDir "stage_a_mu.pth"
            $env:CLOT_PHI_ANCHOR_DIR = "outputs/biochem/anchors_teacher_species"
            # Stage A
            $envA = @{
                CLOT_PHI_REGRESSION_ONLY = "1"
                CLOT_PHI_MU_CAP_SI = "10"
                CLOT_PHI_MU_SOLID_SI = "10"
                CLOT_PHI_JOINT_BIO = "0"
                CLOT_PHI_PHYSICS_BLEND = "0"
            }
            Invoke-ClotTrain -Leg "${leg}_A" -Env $envA -CkptOut $stageA -Epochs 25
            # Stage B
            $envB = @{
                CLOT_PHI_REGRESSION_ONLY = "0"
                CLOT_PHI_MU_CAP_SI = "0.10"
                CLOT_PHI_MU_SOLID_SI = "0.10"
                CLOT_PHI_THRESH_SI = "0.045"
                CLOT_PHI_JOINT_BIO = "1"
                CLOT_PHI_ANCHOR_BALANCED = "1"
                CLOT_PHI_BIO_FI_WEIGHT = "2.0"
                CLOT_PHI_BIO_MAT_WEIGHT = "2.0"
                CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
                CLOT_PHI_PHYSICS_BLEND = "1"
                CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
                CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
                CLOT_PHI_PHYSICS_GELATION_GATE = "1"
                CLOT_PHI_INIT_CHECKPOINT = $stageA
                CLOT_PHI_FREEZE_MU_BRANCH = "1"
            }
            Invoke-ClotTrain -Leg $leg -Env $envB -CkptOut $ckpt -Epochs $ClotEpochs
            Invoke-MultiEval -Ckpt $ckpt -JsonOut $evalJson
        }
        default { Write-Host "[WARN] unknown leg $leg" -ForegroundColor Yellow }
    }

    $s = Summarize-Jsonl $evalJson
    if ($s) {
        $row = [ordered]@{ leg = $leg }
        $s.PSObject.Properties | ForEach-Object { $row[$_.Name] = $_.Value }
        $summary += [pscustomobject]$row
        Write-Host ("[i]  $leg : mean_f1=$($s.mean_f1) min_f1=$($s.min_f1) mean_logMAE=$($s.mean_logMAE) mean_score=$($s.mean_score)") -ForegroundColor Yellow
    }
}

if ($summary.Count -gt 0) {
    $summary | Format-Table -AutoSize
    $summary | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $OutRoot "summary.json") -Encoding utf8
    Write-Host "[OK]  wrote $OutRoot\summary.json" -ForegroundColor Green
}

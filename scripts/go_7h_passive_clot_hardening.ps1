# 7h passive + clot-phi hardening run (autonomous ladder toward full biochem teacher quality).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_7h_passive_clot_hardening.ps1
#
# Steps:
#   0) log baseline
#   1) FI/Mat sweep on adaptive anchor cache
#   2) passive teacher (clot_band) 14 ep
#   3) species dump --min-steps 8
#   4) clot-phi train + multi-anchor
#   5) branch: staged (if min_f1>=gate) else threshold eval sweep

param(
    [switch] $SkipSweep,
    [switch] $SkipTeacher,
    [int] $SweepEpochs = 12,
    [int] $TeacherEpochs = 14,
    [int] $ClotEpochs = 35,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 8,
    [double] $MinF1Gate = 0.38
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutRoot = Join-Path $RepoRoot "outputs\biochem\7h_hardening"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "run_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

$BaselineAnchorDir = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt"
$AnchorDir = Join-Path $OutRoot "anchors_teacher_minsteps8"
$SweepRoot = Join-Path $OutRoot "sweep_fimat"

function Write-RunLog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        step = $Step
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "WARN") { "Yellow" } else { "Cyan" })
}

function Summarize-MultiAnchor {
    param([string] $JsonlPath)
    if (-not (Test-Path $JsonlPath)) { return $null }
    $rows = Get-Content $JsonlPath | ForEach-Object { $_ | ConvertFrom-Json }
    if (-not $rows) { return $null }
    $f1 = @($rows | ForEach-Object { [double]$_.val.clot_f1 })
    $mae = @($rows | ForEach-Object { [double]$_.val.mu_log_mae })
    return [pscustomobject]@{
        mean_f1 = [math]::Round(($f1 | Measure-Object -Average).Average, 3)
        min_f1 = [math]::Round(($f1 | Measure-Object -Minimum).Minimum, 3)
        mean_logMAE = [math]::Round(($mae | Measure-Object -Average).Average, 3)
        path = $JsonlPath
    }
}

function Invoke-PassiveTeacher {
    Write-RunLog "teacher_train" "START" @{ epochs = $TeacherEpochs }
    $env:KINEMATICS_USE_HARD_BCS = "1"
    $env:KINEMATICS_USE_WIDTH_PRIORS = "1"
    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = "7h_passive_clotband_teacher"
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "4"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.25"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"

    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $TeacherEpochs --save-best --run-name 7h_passive_clotband_teacher
    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "teacher_train" "ERR" @{ exit = $LASTEXITCODE }
        exit $LASTEXITCODE
    }
    Write-RunLog "teacher_train" "OK" @{ ckpt = "outputs/biochem/biochem_teacher_last.pth" }
}

# --- Step 0: baseline ---
Write-RunLog "baseline" "START" @{}
$baseEval = "outputs/biochem/passive_species_focus_compare/clotband_adapt_balfi2/multi_anchor.jsonl"
$baseSum = Summarize-MultiAnchor $baseEval
if ($baseSum) {
    Write-RunLog "baseline" "OK" @{
        mean_f1 = $baseSum.mean_f1
        min_f1 = $baseSum.min_f1
        mean_logMAE = $baseSum.mean_logMAE
        ref = $baseEval
    }
} else {
    Write-RunLog "baseline" "WARN" @{ note = "no prior balfi2 eval found" }
}

# --- Step 1: FI/Mat sweep ---
$bestFi = 2.0
$bestMat = 2.0
$bestSweepMinF1 = -1.0
$stagedSum = $null
$finalSum = $null

if (-not $SkipSweep) {
    if (-not (Test-Path (Join-Path $RepoRoot $BaselineAnchorDir))) {
        Write-RunLog "sweep" "ERR" @{ note = "missing anchor dir $BaselineAnchorDir" }
        exit 1
    }
    New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null
    $combos = @(
        @{ fi = 1.5; mat = 1.5; leg = "fi15_mat15" },
        @{ fi = 2.0; mat = 2.0; leg = "fi20_mat20" },
        @{ fi = 3.0; mat = 2.0; leg = "fi30_mat20" },
        @{ fi = 2.0; mat = 3.0; leg = "fi20_mat30" }
    )
    foreach ($c in $combos) {
        Write-RunLog "sweep_$($c.leg)" "START" @{ fi = $c.fi; mat = $c.mat; epochs = $SweepEpochs }
        powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
            -AnchorDir $BaselineAnchorDir `
            -LegName $c.leg `
            -Epochs $SweepEpochs `
            -BioFiWeight $c.fi `
            -BioMatWeight $c.mat
        if ($LASTEXITCODE -ne 0) {
            Write-RunLog "sweep_$($c.leg)" "ERR" @{ exit = $LASTEXITCODE }
            continue
        }
        $evalPath = "outputs/biochem/passive_species_focus_compare/$($c.leg)/multi_anchor.jsonl"
        $s = Summarize-MultiAnchor $evalPath
        if ($s) {
            Write-RunLog "sweep_$($c.leg)" "OK" @{
                mean_f1 = $s.mean_f1
                min_f1 = $s.min_f1
                mean_logMAE = $s.mean_logMAE
            }
            if ($s.min_f1 -gt $bestSweepMinF1 -or ($s.min_f1 -eq $bestSweepMinF1 -and $s.mean_f1 -gt 0)) {
                $bestSweepMinF1 = $s.min_f1
                $bestFi = $c.fi
                $bestMat = $c.mat
            }
        }
    }
    Write-RunLog "sweep_winner" "OK" @{ fi = $bestFi; mat = $bestMat; min_f1 = $bestSweepMinF1 }
} else {
    Write-RunLog "sweep" "SKIP" @{ fi = $bestFi; mat = $bestMat }
}

# --- Step 2: teacher ---
if (-not $SkipTeacher) {
    Invoke-PassiveTeacher
} else {
    Write-RunLog "teacher_train" "SKIP" @{}
}

# --- Step 3: dump ---
Write-RunLog "species_dump" "START" @{ stride = $DumpStride; min_steps = $DumpMinSteps }
python scripts/dump_teacher_species_to_anchors.py `
    --teacher outputs/biochem/biochem_teacher_last.pth `
    --out-dir $AnchorDir `
    --device cuda `
    --time-stride $DumpStride `
    --min-steps $DumpMinSteps `
    --force
if ($LASTEXITCODE -ne 0) {
    Write-RunLog "species_dump" "ERR" @{ exit = $LASTEXITCODE }
    exit $LASTEXITCODE
}
$tinfo = python -c "import glob,torch,os; p=r'$AnchorDir'; print(';'.join([f'{os.path.basename(f)}:{torch.load(f,weights_only=False).y.shape[0]}' for f in sorted(glob.glob(p+'/*.pt'))]))"
Write-RunLog "species_dump" "OK" @{ anchor_T = $tinfo }

# --- Step 4: clot-phi final ---
$FinalLeg = "7h_final"
Write-RunLog "clot_phi_final" "START" @{ leg = $FinalLeg; epochs = $ClotEpochs; fi = $bestFi; mat = $bestMat }
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
    -AnchorDir $AnchorDir `
    -LegName $FinalLeg `
    -Epochs $ClotEpochs `
    -BioFiWeight $bestFi `
    -BioMatWeight $bestMat
if ($LASTEXITCODE -ne 0) {
    Write-RunLog "clot_phi_final" "ERR" @{ exit = $LASTEXITCODE }
    exit $LASTEXITCODE
}
$finalEval = "outputs/biochem/passive_species_focus_compare/$FinalLeg/multi_anchor.jsonl"
$finalSum = Summarize-MultiAnchor $finalEval
Write-RunLog "clot_phi_final" "OK" @{
    mean_f1 = $finalSum.mean_f1
    min_f1 = $finalSum.min_f1
    mean_logMAE = $finalSum.mean_logMAE
    eval = $finalEval
}

# --- Step 5: branch ---
$branch = "threshold_sweep"
if ($finalSum -and $finalSum.min_f1 -ge $MinF1Gate) {
    $branch = "staged"
    Write-RunLog "branch_staged" "START" @{ min_f1 = $finalSum.min_f1; gate = $MinF1Gate }
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $env:CLOT_PHI_ANCHOR_BALANCED = "1"
    $env:CLOT_PHI_BIO_FI_WEIGHT = "$bestFi"
    $env:CLOT_PHI_BIO_MAT_WEIGHT = "$bestMat"
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"
    $StageA = Join-Path $OutRoot "stage_a_mu.pth"
    $StageLeg = Join-Path $OutRoot "staged"
    New-Item -ItemType Directory -Force -Path $StageLeg | Out-Null

    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_REGRESSION_ONLY = "1"
    $env:CLOT_PHI_MODEL = "mlp"
    $env:CLOT_PHI_HYBRID = "1"
    $env:CLOT_PHI_SOFT_LABELS = "0"
    $env:CLOT_PHI_BALANCED = "0"
    $env:CLOT_PHI_DICE_LAMBDA = "0"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "2.0"
    $env:CLOT_PHI_HIDDEN = "32"
    $env:CLOT_PHI_MLP_DEPTH = "2"
    $env:CLOT_PHI_DROPOUT = "0.15"
    $env:CLOT_PHI_LR = "1e-3"
    $env:CLOT_PHI_EPOCHS = "25"
    $env:CLOT_PHI_MU_CAP_SI = "10"
    $env:CLOT_PHI_MU_SOLID_SI = "10"
    $env:CLOT_PHI_JOINT_BIO = "0"
    $env:CLOT_PHI_PHYSICS_BLEND = "0"
    $env:CLOT_PHI_SWEEP_DIR = "outputs/biochem/passive_species_focus_compare"
    $env:CLOT_PHI_SWEEP_LEG = "7h_stage_a"
    Remove-Item Env:CLOT_PHI_INIT_CHECKPOINT, Env:CLOT_PHI_FREEZE_MU_BRANCH -ErrorAction SilentlyContinue
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { Write-RunLog "branch_staged" "ERR" @{ phase = "A"; exit = $LASTEXITCODE }; exit $LASTEXITCODE }
    Copy-Item -Force "outputs/biochem/passive_species_focus_compare/7h_stage_a/clot_phi_best.pth" $StageA

    $env:CLOT_PHI_REGRESSION_ONLY = "0"
    $env:CLOT_PHI_SOFT_LABELS = "1"
    $env:CLOT_PHI_BALANCED = "1"
    $env:CLOT_PHI_DICE_LAMBDA = "0.2"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
    $env:CLOT_PHI_MU_CAP_SI = "0.10"
    $env:CLOT_PHI_MU_SOLID_SI = "0.10"
    $env:CLOT_PHI_THRESH_SI = "0.045"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "1"
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.75"
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_INIT_CHECKPOINT = $StageA
    $env:CLOT_PHI_FREEZE_MU_BRANCH = "1"
    $env:CLOT_PHI_EPOCHS = "25"
    $env:CLOT_PHI_SWEEP_LEG = "7h_staged"
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { Write-RunLog "branch_staged" "ERR" @{ phase = "B"; exit = $LASTEXITCODE }; exit $LASTEXITCODE }
    $stagedCkpt = "outputs/biochem/passive_species_focus_compare/7h_staged/clot_phi_best.pth"
    $stagedEval = Join-Path $OutRoot "staged_multi_anchor.jsonl"
    python scripts/eval_clot_phi_multi_anchor.py --checkpoint $stagedCkpt --out $stagedEval
    $stagedSum = Summarize-MultiAnchor $stagedEval
    Write-RunLog "branch_staged" "OK" @{
        mean_f1 = $stagedSum.mean_f1
        min_f1 = $stagedSum.min_f1
        mean_logMAE = $stagedSum.mean_logMAE
        eval = $stagedEval
    }
} else {
    Write-RunLog "branch_threshold" "START" @{ min_f1 = $finalSum.min_f1; gate = $MinF1Gate }
    $thrRoot = Join-Path $OutRoot "threshold_sweep"
    New-Item -ItemType Directory -Force -Path $thrRoot | Out-Null
    $finalCkpt = "outputs/biochem/passive_species_focus_compare/$FinalLeg/clot_phi_best.pth"
    foreach ($thr in @("0.040", "0.045", "0.050")) {
        $env:CLOT_PHI_THRESH_SI = $thr
        $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
        $thrTag = "thr_" + ($thr -replace '\.', '')
        $outThr = Join-Path $thrRoot ($thrTag + "_multi_anchor.jsonl")
        python scripts/eval_clot_phi_multi_anchor.py --checkpoint $finalCkpt --out $outThr
        $ts = Summarize-MultiAnchor $outThr
        Write-RunLog "threshold_$thr" "OK" @{
            mean_f1 = $ts.mean_f1
            min_f1 = $ts.min_f1
            mean_logMAE = $ts.mean_logMAE
        }
    }
}

$summary = [ordered]@{
    completed_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    best_fi_weight = $bestFi
    best_mat_weight = $bestMat
    final = $finalSum
    branch = $branch
    anchor_dir = $AnchorDir
    log = $LogPath
}
if ($stagedSum) { $summary.staged = $stagedSum }
$summary | ConvertTo-Json -Depth 5 | Set-Content $SummaryPath -Encoding utf8
Write-Host "[OK] 7h hardening complete -> $SummaryPath" -ForegroundColor Green
exit 0

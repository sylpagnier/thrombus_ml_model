# GT-flow species ladder (~6h budget): smoke -> clot-band teacher -> dump -> clot-phi legs.
# No Stage-A kinematics checkpoint required (BIOCHEM_GT_KINE_VEL=1).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_species_ladder_6h.ps1"
#   powershell ... -SkipPytest -SkipSmoke -TeacherEpochs 12 -ClotEpochs 28

param(
    [switch] $SkipPytest,
    [switch] $SkipSmoke,
    [switch] $SkipTeacher,
    [switch] $SkipDump,
    [int] $SmokeEpochs = 2,
    [int] $TeacherEpochs = 12,
    [int] $ClotEpochs = 28,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 6,
    [double] $MinF1Promote = 0.34
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_passive_gt_flow_env.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_ladder_6h"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "ladder_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

function Write-LadderLog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        step = $Step
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } elseif ($Status -eq "WARN") { "Yellow" } else { "Cyan" })
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

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs)
    # Python may write UserWarnings to stderr; do not treat as terminating errors.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -u @PyArgs
    $rc = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    $ErrorActionPreference = $prevEap
    return [int]$rc
}

function Invoke-TeacherTrain {
    param([string] $RunNote, [int] $Epochs, [switch] $ClotBand, [switch] $AdrBackprop)
    Set-PassiveGtFlowEnv -RunNote $RunNote -ClotBandMask:$ClotBand -AdrBackprop:$AdrBackprop
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $logFile = Join-Path $OutRoot ("train_" + $RunNote + ".log")
    Write-LadderLog "teacher_$RunNote" "START" @{ epochs = $Epochs; log = $logFile }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $Epochs --save-best --run-name $RunNote 2>&1 | Tee-Object -FilePath $logFile
    $trainRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($trainRc -ne 0) {
        Write-LadderLog "teacher_$RunNote" "FAIL" @{ exit = $trainRc }
        exit $trainRc
    }
    $gateRc = Invoke-Python @("scripts/check_passive_teacher_gate.py", "--run-note", $RunNote)
    Write-LadderLog "teacher_$RunNote" $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ gate_exit = $gateRc }
    if ($gateRc -ne 0) {
        Write-Host "[WARN] teacher gate failed for $RunNote; continuing ladder (review log)" -ForegroundColor Yellow
    }
}

# --- Step 0: physics unit tests ---
if (-not $SkipPytest) {
    Write-LadderLog "pytest_biochem_physics" "START" @{}
    python -m pytest src/tests/test_biochem_physics.py -q --tb=short
    if ($LASTEXITCODE -ne 0) {
        Write-LadderLog "pytest_biochem_physics" "FAIL" @{ exit = $LASTEXITCODE }
        exit $LASTEXITCODE
    }
    Write-LadderLog "pytest_biochem_physics" "OK" @{}
}

# --- Step 1: smoke (clot-band, ADR off) ---
if (-not $SkipSmoke) {
    Invoke-TeacherTrain -RunNote "ladder_smoke_cb" -Epochs $SmokeEpochs -ClotBand
}

# --- Step 2: main clot-band teacher (init from phaseB / best) ---
$TeacherCkpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
if (-not $SkipTeacher) {
    if (Test-Path $TeacherCkpt) {
        Copy-Item $TeacherCkpt (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
        Write-Host "[i] teacher init: $TeacherCkpt -> biochem_teacher_best_high_mu.pth" -ForegroundColor Cyan
    }
    Invoke-TeacherTrain -RunNote "ladder_cb_teacher" -Epochs $TeacherEpochs -ClotBand
    Copy-Item $TeacherCkpt (Join-Path $OutRoot "biochem_teacher_ladder_last.pth") -Force
}

if (-not (Test-Path $TeacherCkpt)) {
    Write-LadderLog "teacher_ckpt" "FAIL" @{ path = $TeacherCkpt }
    exit 1
}

# --- Step 3: dump species ---
$AnchorDir = Join-Path $OutRoot ("anchors_stride" + $DumpStride + "_m" + $DumpMinSteps)
if (-not $SkipDump) {
    Write-LadderLog "species_dump" "START" @{ out = $AnchorDir }
    $dumpRc = Invoke-Python scripts/dump_teacher_species_to_anchors.py `
        --teacher $TeacherCkpt --out-dir $AnchorDir --device cuda `
        --time-stride $DumpStride --min-steps $DumpMinSteps --force
    if ($dumpRc -ne 0) {
        Write-LadderLog "species_dump" "FAIL" @{ exit = $dumpRc }
        exit $dumpRc
    }
    Write-LadderLog "species_dump" "OK" @{ out = $AnchorDir }
}

# --- Step 4: clot-phi legs on fresh dump ---
$clotSummary = @()
$legs = @(
    @{
        name = "gtsp_blend"
        pred_species = "0"
        fi = 2.0
        mat = 2.0
        alpha = "0.55"
    },
    @{
        name = "recovery_fi30"
        pred_species = "1"
        fi = 3.0
        mat = 2.0
        alpha = "0.75"
    }
)

foreach ($leg in $legs) {
    $legDir = Join-Path $OutRoot $leg.name
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    Write-LadderLog ("clot_" + $leg.name) "START" @{ anchor_dir = $AnchorDir; epochs = $ClotEpochs }

    Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
        Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
    }
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $env:CLOT_PHI_EPOCHS = "$ClotEpochs"
    $env:CLOT_PHI_LR = "1e-3"
    $env:CLOT_PHI_WEIGHT_DECAY = "1e-4"
    $env:CLOT_PHI_HYBRID = "1"
    $env:CLOT_PHI_SOFT_LABELS = "1"
    $env:CLOT_PHI_BALANCED = "1"
    $env:CLOT_PHI_POS_WEIGHT_CAP = "8"
    $env:CLOT_PHI_MINIMAL_FEATURES = "1"
    $env:CLOT_PHI_SPECIES_FEATURES = "0"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_ANCHOR_BALANCED = "1"
    $env:CLOT_PHI_BIO_FI_WEIGHT = "$($leg.fi)"
    $env:CLOT_PHI_BIO_MAT_WEIGHT = "$($leg.mat)"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = $leg.pred_species
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = $leg.alpha
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
    $env:CLOT_PHI_DICE_LAMBDA = "0.2"
    $env:CLOT_PHI_MODEL = "mlp"
    $env:CLOT_PHI_HIDDEN = "32"
    $env:CLOT_PHI_MLP_DEPTH = "2"
    $env:CLOT_PHI_DROPOUT = "0.15"
    $env:CLOT_PHI_THRESH_SI = "0.045"
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"
    $env:CLOT_PHI_SWEEP_DIR = $OutRoot
    $env:CLOT_PHI_SWEEP_LEG = $leg.name

    $clotRc = Invoke-Python -m src.training.train_clot_phi_simple
    if ($clotRc -ne 0) {
        Write-LadderLog ("clot_" + $leg.name) "FAIL" @{ exit = $clotRc }
        exit $clotRc
    }

    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    if (-not (Test-Path $ckpt)) {
        $ckpt = Join-Path $OutRoot ($leg.name + "/clot_phi_best.pth")
    }
    $evalOut = Join-Path $legDir "multi_anchor.jsonl"
    $evalRc = Invoke-Python scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut
    if ($evalRc -ne 0) {
        Write-LadderLog ("clot_" + $leg.name) "FAIL" @{ exit = $evalRc; phase = "eval" }
        exit $evalRc
    }

    $sum = Summarize-MultiAnchor $evalOut
    if ($sum) {
        Write-LadderLog ("clot_" + $leg.name) "OK" @{
            mean_f1 = $sum.mean_f1
            min_f1 = $sum.min_f1
            mean_logMAE = $sum.mean_logMAE
        }
        $clotSummary += [pscustomobject]@{
            leg = $leg.name
            mean_f1 = $sum.mean_f1
            min_f1 = $sum.min_f1
            mean_logMAE = $sum.mean_logMAE
            eval = $evalOut
        }
    }
}

# --- Promote best leg ---
$bestLeg = $null
$bestMinF1 = -1.0
foreach ($row in $clotSummary) {
    if ([double]$row.min_f1 -gt $bestMinF1) {
        $bestMinF1 = [double]$row.min_f1
        $bestLeg = $row.leg
    }
}

$promoteDir = Join-Path $OutRoot "promoted"
if ($bestLeg) {
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    $srcCkpt = Join-Path $OutRoot "$bestLeg\clot_phi_best.pth"
    Copy-Item $srcCkpt (Join-Path $promoteDir "clot_phi_best.pth") -Force
    Copy-Item (Join-Path $OutRoot "$bestLeg\multi_anchor.jsonl") (Join-Path $promoteDir "multi_anchor.jsonl") -Force
    Write-LadderLog "promote" "OK" @{ leg = $bestLeg; min_f1 = $bestMinF1; gate = $MinF1Promote }
} else {
    Write-LadderLog "promote" "WARN" @{ note = "no clot summary" }
}

if ($clotSummary.Count -gt 0) {
    $clotSummary | Format-Table -AutoSize
    @{
        teacher_ckpt = $TeacherCkpt
        anchor_dir = $AnchorDir
        clot_legs = $clotSummary
        promoted_leg = $bestLeg
        promoted_min_f1 = $bestMinF1
        min_f1_gate = $MinF1Promote
        beat_gate = ($bestMinF1 -ge $MinF1Promote)
    } | ConvertTo-Json -Depth 5 | Set-Content $SummaryPath -Encoding utf8
    Write-Host "[OK] summary: $SummaryPath" -ForegroundColor Green
}

Write-Host "[OK] GT-flow species ladder complete" -ForegroundColor Green

# GT-flow 8h queue (no Stage-A kin model): harden teacher -> dump -> FI/Mat sweep -> long clot-phi.
# Continues from outputs/biochem/gt_flow_ladder_6h/ when present.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_queue_8h.ps1"
#   ... -SkipTeacher -SkipDump   # clot sweep only on existing anchors

param(
    [switch] $SkipTeacher,
    [switch] $SkipDump,
    [switch] $SkipSweep,
    [switch] $SkipAdrRamp,
    [int] $TeacherEpochs = 16,
    [int] $SweepEpochs = 22,
    [int] $FinalEpochs = 45,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 8,
    [double] $MinF1Gate = 0.34
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_passive_gt_flow_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_queue_8h"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "queue_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

$LadderRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_ladder_6h"
$LadderTeacher = Join-Path $LadderRoot "biochem_teacher_ladder_last.pth"
$AnchorDir = Join-Path $OutRoot ("anchors_stride" + $DumpStride + "_m" + $DumpMinSteps)
$SweepRoot = Join-Path $OutRoot "sweep_fimat"
$AdaptBaseline = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt"

function Write-QueueLog {
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

function Invoke-TeacherTrain {
    param([string] $RunNote, [int] $Epochs)
    Set-PassiveGtFlowEnv -RunNote $RunNote -ClotBandMask -AdrBackprop:$false
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $logFile = Join-Path $OutRoot ("train_" + $RunNote + ".log")
    Write-QueueLog "teacher_$RunNote" "START" @{ epochs = $Epochs; log = $logFile }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -u -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs $Epochs --save-best --run-name $RunNote 2>&1 | Tee-Object -FilePath $logFile
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($rc -ne 0) {
        Write-QueueLog "teacher_$RunNote" "FAIL" @{ exit = $rc }
        exit $rc
    }
    Invoke-PythonRcRc scripts/check_passive_teacher_gate.py --run-note $RunNote -Quiet | Out-Null
    $gateRc = $LASTEXITCODE
    Write-QueueLog "teacher_$RunNote" $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ gate_exit = $gateRc }
}

# --- Baseline: prior ladder + adapt cache eval ---
Write-QueueLog "baseline_ladder" "START" @{}
$ladderSum = Summarize-MultiAnchor (Join-Path $LadderRoot "promoted\multi_anchor.jsonl")
if ($ladderSum) {
    Write-QueueLog "baseline_ladder" "OK" @{
        mean_f1 = $ladderSum.mean_f1
        min_f1 = $ladderSum.min_f1
        ref = "gt_flow_ladder_6h/promoted"
    }
} else {
    Write-QueueLog "baseline_ladder" "WARN" @{ note = "no ladder promoted eval" }
}

$adaptEval = Join-Path $OutRoot "baseline_adapt_fi30_eval.jsonl"
if (Test-Path (Join-Path $RepoRoot $AdaptBaseline)) {
    $adaptCkpt = "outputs/biochem/passive_species_focus_compare/recovery_adapt_fi30/clot_phi_best.pth"
    if (Test-Path (Join-Path $RepoRoot $adaptCkpt)) {
        Write-QueueLog "baseline_adapt_ckpt" "START" @{ ckpt = $adaptCkpt }
        $rc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $adaptCkpt --out $adaptEval
        $s = Summarize-MultiAnchor $adaptEval
        if ($s -and $rc -eq 0) {
            Write-QueueLog "baseline_adapt_ckpt" "OK" @{
                mean_f1 = $s.mean_f1
                min_f1 = $s.min_f1
                mean_logMAE = $s.mean_logMAE
            }
        }
    }
}

# --- Teacher (init from ladder teacher if present) ---
$TeacherCkpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
if (-not $SkipTeacher) {
    if (Test-Path $LadderTeacher) {
        Copy-Item $LadderTeacher (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
        Write-Host "[i] init from ladder teacher: $LadderTeacher" -ForegroundColor Cyan
    }
    Invoke-TeacherTrain -RunNote "queue_cb_teacher_16ep" -Epochs $TeacherEpochs
    Copy-Item $TeacherCkpt (Join-Path $OutRoot "biochem_teacher_queue_last.pth") -Force
}

if (-not (Test-Path $TeacherCkpt)) {
    Write-QueueLog "teacher_ckpt" "FAIL" @{ path = $TeacherCkpt }
    exit 1
}

# --- Dump ---
if (-not $SkipDump) {
    Write-QueueLog "species_dump" "START" @{ out = $AnchorDir; min_steps = $DumpMinSteps }
    $dumpRc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $TeacherCkpt --out-dir $AnchorDir --device cuda `
        --time-stride $DumpStride --min-steps $DumpMinSteps --force
    if ($dumpRc -ne 0) {
        Write-QueueLog "species_dump" "FAIL" @{ exit = $dumpRc }
        exit $dumpRc
    }
    Write-QueueLog "species_dump" "OK" @{ out = $AnchorDir }
}

if (-not (Test-Path $AnchorDir)) {
    Write-QueueLog "anchor_dir" "FAIL" @{ path = $AnchorDir }
    exit 1
}

# --- FI/Mat sweep on fresh dump ---
$bestFi = 3.0
$bestMat = 2.0
$bestMinF1 = -1.0
$sweepRows = @()

if (-not $SkipSweep) {
    New-Item -ItemType Directory -Force -Path $SweepRoot | Out-Null
    $combos = @(
        @{ fi = 2.0; mat = 2.0; leg = "fi20_mat20" },
        @{ fi = 3.0; mat = 2.0; leg = "fi30_mat20" },
        @{ fi = 2.0; mat = 3.0; leg = "fi20_mat30" },
        @{ fi = 3.0; mat = 3.0; leg = "fi30_mat30" }
    )
    foreach ($c in $combos) {
        Write-QueueLog ("sweep_" + $c.leg) "START" @{ fi = $c.fi; mat = $c.mat; epochs = $SweepEpochs }
        $legDir = Join-Path $SweepRoot $c.leg
        New-Item -ItemType Directory -Force -Path $legDir | Out-Null

        Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
            Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
        }
        . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
        $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
        $env:CLOT_PHI_EPOCHS = "$SweepEpochs"
        $env:CLOT_PHI_JOINT_BIO = "1"
        $env:CLOT_PHI_BIO_LAMBDA = "0.25"
        $env:CLOT_PHI_ANCHOR_BALANCED = "1"
        $env:CLOT_PHI_BIO_FI_WEIGHT = "$($c.fi)"
        $env:CLOT_PHI_BIO_MAT_WEIGHT = "$($c.mat)"
        $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
        $env:CLOT_PHI_PHYSICS_BLEND = "1"
        $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = "0.55"
        $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
        $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
        $env:CLOT_PHI_SWEEP_DIR = $SweepRoot
        $env:CLOT_PHI_SWEEP_LEG = $c.leg
        $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"

        $rc = Invoke-PythonRcRc -m src.training.train_clot_phi_simple
        if ($rc -ne 0) {
            Write-QueueLog ("sweep_" + $c.leg) "FAIL" @{ exit = $rc }
            continue
        }
        $ckpt = Join-Path $legDir "clot_phi_best.pth"
        $evalOut = Join-Path $legDir "multi_anchor.jsonl"
        $erc = Invoke-PythonRcRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut -Quiet
        if ($erc -ne 0) {
            Write-QueueLog ("sweep_" + $c.leg) "FAIL" @{ exit = $erc; phase = "eval" }
            continue
        }
        $s = Summarize-MultiAnchor $evalOut
        if ($s) {
            Write-QueueLog ("sweep_" + $c.leg) "OK" @{
                mean_f1 = $s.mean_f1
                min_f1 = $s.min_f1
                mean_logMAE = $s.mean_logMAE
            }
            $sweepRows += [pscustomobject]@{ leg = $c.leg; fi = $c.fi; mat = $c.mat; min_f1 = $s.min_f1; mean_f1 = $s.mean_f1 }
            if ($s.min_f1 -gt $bestMinF1) {
                $bestMinF1 = $s.min_f1
                $bestFi = $c.fi
                $bestMat = $c.mat
            }
        }
    }
    Write-QueueLog "sweep_winner" "OK" @{ fi = $bestFi; mat = $bestMat; min_f1 = $bestMinF1 }
}

# --- Final long clot legs (gtsp + pred species) with sweep winner weights ---
$finalRows = @()
$finalLegs = @(
    @{ name = "final_gtsp"; pred = "0"; alpha = "0.55" },
    @{ name = "final_blend"; pred = "1"; alpha = "0.75" }
)
foreach ($fl in $finalLegs) {
    Write-QueueLog ("clot_" + $fl.name) "START" @{ epochs = $FinalEpochs; fi = $bestFi; mat = $bestMat }
    $legDir = Join-Path $OutRoot $fl.name
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null

    Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
        Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
    }
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $env:CLOT_PHI_EPOCHS = "$FinalEpochs"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_ANCHOR_BALANCED = "1"
    $env:CLOT_PHI_BIO_FI_WEIGHT = "$bestFi"
    $env:CLOT_PHI_BIO_MAT_WEIGHT = "$bestMat"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = $fl.pred
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = $fl.alpha
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_SWEEP_DIR = $OutRoot
    $env:CLOT_PHI_SWEEP_LEG = $fl.name
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"

    $rc = Invoke-PythonRc -m src.training.train_clot_phi_simple
    if ($rc -ne 0) {
        Write-QueueLog ("clot_" + $fl.name) "FAIL" @{ exit = $rc }
        continue
    }
    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    $evalOut = Join-Path $legDir "multi_anchor.jsonl"
    $erc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut
    if ($erc -ne 0) {
        Write-QueueLog ("clot_" + $fl.name) "FAIL" @{ exit = $erc; phase = "eval" }
        continue
    }
    $s = Summarize-MultiAnchor $evalOut
    if ($s) {
        Write-QueueLog ("clot_" + $fl.name) "OK" @{
            mean_f1 = $s.mean_f1
            min_f1 = $s.min_f1
            mean_logMAE = $s.mean_logMAE
        }
        $finalRows += [pscustomobject]@{ leg = $fl.name; min_f1 = $s.min_f1; mean_f1 = $s.mean_f1; mean_logMAE = $s.mean_logMAE }
    }
}

# --- Optional: Phase B ADR ramp on queue teacher ---
if (-not $SkipAdrRamp) {
    Write-QueueLog "phaseB_adr_ramp" "START" @{ epochs = 8 }
    Copy-Item $TeacherCkpt (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
    Set-PassiveGtFlowEnv -RunNote "queue_adr_ramp2" -ClotBandMask -AdrBackprop -AdrWeight "1e-3"
    $env:BIOCHEM_TEACHER_EPOCHS = "8"
    $env:BIOCHEM_EPOCHS = "8"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "8"
    $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best --epochs 8 --save-best --run-name queue_adr_ramp2
    Write-QueueLog "phaseB_adr_ramp" $(if ($rc -eq 0) { "OK" } else { "WARN" }) @{ exit = $rc }
    if ($rc -eq 0) {
        $adrAnchor = Join-Path $OutRoot "anchors_adr_stride36_m6"
        $drc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
            --teacher $TeacherCkpt --out-dir $adrAnchor --device cuda `
            --time-stride 36 --min-steps 6 --force
        if ($drc -eq 0) {
            Write-QueueLog "adr_redump" "OK" @{ out = $adrAnchor }
            $adrLeg = Join-Path $OutRoot "adr_final_gtsp"
            New-Item -ItemType Directory -Force -Path $adrLeg | Out-Null
            Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
                Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
            }
            . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
            $env:CLOT_PHI_ANCHOR_DIR = $adrAnchor
            $env:CLOT_PHI_EPOCHS = "35"
            $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = "0"
            $env:CLOT_PHI_BIO_FI_WEIGHT = "$bestFi"
            $env:CLOT_PHI_BIO_MAT_WEIGHT = "$bestMat"
            $env:CLOT_PHI_SWEEP_DIR = $OutRoot
            $env:CLOT_PHI_SWEEP_LEG = "adr_final_gtsp"
            $trc = Invoke-PythonRc -m src.training.train_clot_phi_simple
            if ($trc -eq 0) {
                $ck = Join-Path $adrLeg "clot_phi_best.pth"
                $ev = Join-Path $adrLeg "multi_anchor.jsonl"
                Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ck --out $ev | Out-Null
                $s = Summarize-MultiAnchor $ev
                if ($s) {
                    Write-QueueLog "adr_clot_eval" "OK" @{ min_f1 = $s.min_f1; mean_f1 = $s.mean_f1 }
                    $finalRows += [pscustomobject]@{ leg = "adr_final_gtsp"; min_f1 = $s.min_f1; mean_f1 = $s.mean_f1; mean_logMAE = $s.mean_logMAE }
                }
            }
        }
    }
}

# --- Promote best overall ---
$promoteDir = Join-Path $OutRoot "promoted"
$bestLeg = $null
$bestMin = -1.0
foreach ($r in $finalRows) {
    if ([double]$r.min_f1 -gt $bestMin) {
        $bestMin = [double]$r.min_f1
        $bestLeg = $r.leg
    }
}
if ($bestLeg) {
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    $src = Join-Path $OutRoot "$bestLeg\clot_phi_best.pth"
    Copy-Item $src (Join-Path $promoteDir "clot_phi_best.pth") -Force
    Copy-Item (Join-Path $OutRoot "$bestLeg\multi_anchor.jsonl") (Join-Path $promoteDir "multi_anchor.jsonl") -Force
    Write-QueueLog "promote" "OK" @{ leg = $bestLeg; min_f1 = $bestMin; gate = $MinF1Gate; beat_gate = ($bestMin -ge $MinF1Gate) }
}

@{
    anchor_dir = $AnchorDir
    sweep = $sweepRows
    finals = $finalRows
    promoted_leg = $bestLeg
    promoted_min_f1 = $bestMin
    min_f1_gate = $MinF1Gate
    ladder_baseline = $ladderSum
} | ConvertTo-Json -Depth 6 | Set-Content $SummaryPath -Encoding utf8

Write-Host "[OK] 8h queue complete -> $SummaryPath" -ForegroundColor Green

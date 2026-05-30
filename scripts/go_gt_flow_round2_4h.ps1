# GT-flow round 2 (~4h): proven anchor caches only (no m8 re-dump).
# Focus: beat min_f1>=0.34 on multi-anchor clot-phi.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_round2_4h.ps1"
#   ... -SkipSweep -SkipLong -SkipTeacherRefresh

param(
    [switch] $SkipThreshold,
    [switch] $SkipSweep,
    [switch] $SkipLong,
    [switch] $SkipTeacherRefresh,
    [int] $SweepEpochs = 28,
    [int] $LongEpochs = 65,
    [int] $TeacherRefreshEpochs = 10,
    [double] $MinF1Gate = 0.34
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_gt_flow_env.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_round2_4h"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "round2_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

$LadderM6 = "outputs/biochem/gt_flow_ladder_6h/anchors_stride36_m6"
$AdaptCache = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt"
$LadderPromotedCkpt = "outputs/biochem/gt_flow_ladder_6h/promoted/clot_phi_best.pth"
$AdaptRecoveryCkpt = "outputs/biochem/passive_species_focus_compare/recovery_adapt_fi30/clot_phi_best.pth"
$TeacherInit = "outputs/biochem/biochem_teacher_last.pth"

function Write-R2Log {
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

function Clear-ClotPhiEnv {
    Get-ChildItem Env: | Where-Object { $_.Name -like "CLOT_PHI_*" } | ForEach-Object {
        Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue
    }
}

function Set-ClotPhiRecipe {
    param(
        [string] $AnchorDir,
        [string] $LegName,
        [string] $OutDir,
        [int] $Epochs,
        [double] $Fi,
        [double] $Mat,
        [string] $PredSpecies,
        [string] $Alpha,
        [string] $ThreshSi = "0.045"
    )
    Clear-ClotPhiEnv
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $env:CLOT_PHI_EPOCHS = "$Epochs"
    $env:CLOT_PHI_THRESH_SI = $ThreshSi
    $env:CLOT_PHI_MODEL = "mlp"
    $env:CLOT_PHI_HIDDEN = "32"
    $env:CLOT_PHI_MLP_DEPTH = "2"
    $env:CLOT_PHI_DROPOUT = "0.15"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "1.5"
    $env:CLOT_PHI_DICE_LAMBDA = "0.2"
    $env:CLOT_PHI_JOINT_BIO = "1"
    $env:CLOT_PHI_BIO_LAMBDA = "0.25"
    $env:CLOT_PHI_ANCHOR_BALANCED = "1"
    $env:CLOT_PHI_BIO_FI_WEIGHT = "$Fi"
    $env:CLOT_PHI_BIO_MAT_WEIGHT = "$Mat"
    $env:CLOT_PHI_JOINT_USE_PRED_SPECIES = $PredSpecies
    $env:CLOT_PHI_PHYSICS_BLEND = "1"
    $env:CLOT_PHI_PHYSICS_BLEND_ALPHA = $Alpha
    $env:CLOT_PHI_PHYSICS_MU_RATIO_MAX = "4"
    $env:CLOT_PHI_PHYSICS_GELATION_GATE = "1"
    $env:CLOT_PHI_TIME_STRIDE_AUTO = "1"
    # train_clot_phi_simple saves to SWEEP_DIR/SWEEP_LEG/clot_phi_best.pth
    $env:CLOT_PHI_SWEEP_DIR = (Split-Path $OutDir -Parent)
    $env:CLOT_PHI_SWEEP_LEG = $LegName
}

function Run-ClotLeg {
    param(
        [string] $StepName,
        [string] $AnchorDir,
        [string] $LegDir,
        [int] $Epochs,
        [double] $Fi,
        [double] $Mat,
        [string] $PredSpecies,
        [string] $Alpha
    )
    New-Item -ItemType Directory -Force -Path $LegDir | Out-Null
    Write-R2Log $StepName "START" @{ anchor = $AnchorDir; epochs = $Epochs; fi = $Fi; mat = $Mat }
    Set-ClotPhiRecipe -AnchorDir $AnchorDir -LegName (Split-Path $LegDir -Leaf) -OutDir $LegDir -Epochs $Epochs `
        -Fi $Fi -Mat $Mat -PredSpecies $PredSpecies -Alpha $Alpha
    $rc = Invoke-PythonRc -m src.training.train_clot_phi_simple
    if ($rc -ne 0) {
        Write-R2Log $StepName "FAIL" @{ exit = $rc; phase = "train" }
        return $null
    }
    $ckpt = Join-Path $LegDir "clot_phi_best.pth"
    $evalOut = Join-Path $LegDir "multi_anchor.jsonl"
    $erc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut -Quiet
    if ($erc -ne 0) {
        Write-R2Log $StepName "FAIL" @{ exit = $erc; phase = "eval" }
        return $null
    }
    $s = Summarize-MultiAnchor $evalOut
    if ($s) {
        Write-R2Log $StepName "OK" @{
            mean_f1 = $s.mean_f1
            min_f1 = $s.min_f1
            mean_logMAE = $s.mean_logMAE
        }
    }
    return $s
}

function Run-ThresholdSweep {
    param([string] $Ckpt, [string] $AnchorDir, [string] $ThrRoot)
    New-Item -ItemType Directory -Force -Path $ThrRoot | Out-Null
    $best = $null
    foreach ($thr in @("0.040", "0.045", "0.050", "0.055")) {
        $env:CLOT_PHI_THRESH_SI = $thr
        $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
        $tag = "thr_" + ($thr -replace '\.', '')
        $out = Join-Path $ThrRoot ($tag + ".jsonl")
        $rc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $out -Quiet
        if ($rc -ne 0) { continue }
        $s = Summarize-MultiAnchor $out
        if ($s) {
            Write-R2Log "threshold_$tag" "OK" @{ min_f1 = $s.min_f1; mean_f1 = $s.mean_f1; thr = $thr }
            if ($null -eq $best -or $s.min_f1 -gt $best.min_f1) { $best = $s }
        }
    }
    return $best
}

# --- Baselines (eval only) ---
$baselineRows = @()
foreach ($b in @(
        @{ name = "ladder_promoted"; ckpt = $LadderPromotedCkpt; anchor = $LadderM6 },
        @{ name = "adapt_recovery"; ckpt = $AdaptRecoveryCkpt; anchor = $AdaptCache }
    )) {
    if (-not (Test-Path (Join-Path $RepoRoot $b.ckpt))) { continue }
    $ev = Join-Path $OutRoot ("baseline_" + $b.name + ".jsonl")
    $env:CLOT_PHI_ANCHOR_DIR = $b.anchor
    $rc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $b.ckpt --out $ev -Quiet
    if ($rc -eq 0) {
        $s = Summarize-MultiAnchor $ev
        if ($s) {
            Write-R2Log ("baseline_" + $b.name) "OK" @{ min_f1 = $s.min_f1; mean_f1 = $s.mean_f1 }
            $baselineRows += [pscustomobject]@{ name = $b.name; min_f1 = $s.min_f1; mean_f1 = $s.mean_f1 }
        }
    }
}

# --- Threshold sweep on existing best ckpts (no retrain) ---
$thrResults = @()
if (-not $SkipThreshold) {
    if (Test-Path (Join-Path $RepoRoot $LadderPromotedCkpt)) {
        $t = Run-ThresholdSweep -Ckpt $LadderPromotedCkpt -AnchorDir $LadderM6 `
            -ThrRoot (Join-Path $OutRoot "threshold_ladder_promoted")
        if ($t) { $thrResults += [pscustomobject]@{ src = "ladder_promoted"; min_f1 = $t.min_f1; mean_f1 = $t.mean_f1 } }
    }
}

# --- FI/Mat sweep on ladder m6 (gtsp / pred off) ---
$bestFi = 3.0
$bestMat = 2.0
$bestMinF1 = -1.0
$sweepRows = @()

if (-not $SkipSweep) {
    if (-not (Test-Path (Join-Path $RepoRoot $LadderM6))) {
        Write-R2Log "sweep" "FAIL" @{ note = "missing $LadderM6" }
        exit 1
    }
    $sweepRoot = Join-Path $OutRoot "sweep_ladder_m6"
    $combos = @(
        @{ fi = 2.0; mat = 2.0; leg = "fi20_mat20" },
        @{ fi = 3.0; mat = 2.0; leg = "fi30_mat20" },
        @{ fi = 2.0; mat = 3.0; leg = "fi20_mat30" },
        @{ fi = 3.0; mat = 3.0; leg = "fi30_mat30" }
    )
    foreach ($c in $combos) {
        $legDir = Join-Path $sweepRoot $c.leg
        $s = Run-ClotLeg -StepName ("sweep_" + $c.leg) -AnchorDir $LadderM6 -LegDir $legDir `
            -Epochs $SweepEpochs -Fi $c.fi -Mat $c.mat -PredSpecies "0" -Alpha "0.55"
        if ($s -and $s.min_f1 -gt $bestMinF1) {
            $bestMinF1 = $s.min_f1
            $bestFi = $c.fi
            $bestMat = $c.mat
        }
        if ($s) {
            $sweepRows += [pscustomobject]@{ leg = $c.leg; min_f1 = $s.min_f1; mean_f1 = $s.mean_f1 }
        }
    }
    Write-R2Log "sweep_winner" "OK" @{ fi = $bestFi; mat = $bestMat; min_f1 = $bestMinF1 }
}

# --- Long clot legs on ladder m6 + adapt cache ---
$longRows = @()
if (-not $SkipLong) {
    $longSpecs = @(
        @{ step = "long_ladder_gtsp"; anchor = $LadderM6; dir = "long_ladder_gtsp"; pred = "0"; alpha = "0.55" },
        @{ step = "long_ladder_blend"; anchor = $LadderM6; dir = "long_ladder_blend"; pred = "1"; alpha = "0.75" },
        @{ step = "long_adapt_blend"; anchor = $AdaptCache; dir = "long_adapt_blend"; pred = "1"; alpha = "0.75" },
        @{ step = "long_adapt_gtsp"; anchor = $AdaptCache; dir = "long_adapt_gtsp"; pred = "0"; alpha = "0.55" }
    )
    foreach ($spec in $longSpecs) {
        if (-not (Test-Path (Join-Path $RepoRoot $spec.anchor))) {
            Write-R2Log $spec.step "WARN" @{ note = "missing anchor dir" }
            continue
        }
        $legDir = Join-Path $OutRoot $spec.dir
        $s = Run-ClotLeg -StepName $spec.step -AnchorDir $spec.anchor -LegDir $legDir `
            -Epochs $LongEpochs -Fi $bestFi -Mat $bestMat -PredSpecies $spec.pred -Alpha $spec.alpha
        if ($s) {
            $longRows += [pscustomobject]@{
                leg = $spec.dir
                anchor = $spec.anchor
                min_f1 = $s.min_f1
                mean_f1 = $s.mean_f1
                mean_logMAE = $s.mean_logMAE
            }
        }
    }
}

# --- Optional: short teacher refresh + ladder m6 re-dump (avoid m8) ---
$refreshAnchor = $LadderM6
if (-not $SkipTeacherRefresh) {
    if (Test-Path (Join-Path $RepoRoot $TeacherInit)) {
        Write-R2Log "teacher_refresh" "START" @{ epochs = $TeacherRefreshEpochs }
        Copy-Item (Join-Path $RepoRoot $TeacherInit) (Join-Path $RepoRoot "outputs/biochem/biochem_teacher_best_high_mu.pth") -Force
        Set-PassiveGtFlowEnv -RunNote "round2_teacher_refresh" -ClotBandMask -AdrBackprop:$false
        $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherRefreshEpochs"
        $env:BIOCHEM_EPOCHS = "$TeacherRefreshEpochs"
        $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherRefreshEpochs"
        $rc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
            --epochs $TeacherRefreshEpochs --save-best --run-name round2_teacher_refresh
        if ($rc -eq 0) {
            $refreshAnchor = Join-Path $OutRoot "anchors_refresh_stride36_m6"
            $drc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
                --teacher outputs/biochem/biochem_teacher_last.pth `
                --out-dir $refreshAnchor --device cuda --time-stride 36 --min-steps 6 --force
            if ($drc -eq 0) {
                Write-R2Log "teacher_refresh_dump" "OK" @{ out = $refreshAnchor }
                $legDir = Join-Path $OutRoot "long_refresh_blend"
                $s = Run-ClotLeg -StepName "long_refresh_blend" -AnchorDir $refreshAnchor -LegDir $legDir `
                    -Epochs $LongEpochs -Fi $bestFi -Mat $bestMat -PredSpecies "1" -Alpha "0.75"
                if ($s) {
                    $longRows += [pscustomobject]@{
                        leg = "long_refresh_blend"
                        anchor = $refreshAnchor
                        min_f1 = $s.min_f1
                        mean_f1 = $s.mean_f1
                    }
                }
            } else {
                Write-R2Log "teacher_refresh_dump" "FAIL" @{ exit = $drc }
            }
        } else {
            Write-R2Log "teacher_refresh" "FAIL" @{ exit = $rc }
        }
    }
}

# --- Threshold on top long ckpts ---
$topCkpts = @()
foreach ($r in $longRows) {
    $ck = Join-Path $OutRoot ($r.leg + "/clot_phi_best.pth")
    if (Test-Path $ck) {
        $topCkpts += @{ name = $r.leg; ckpt = $ck; anchor = $r.anchor }
    }
}
# sort by min_f1 from longRows - take top 2
$sorted = $longRows | Sort-Object -Property @{ Expression = { [double]$_.min_f1 }; Descending = $true } | Select-Object -First 2
foreach ($r in $sorted) {
    $ck = Join-Path $OutRoot ($r.leg + "/clot_phi_best.pth")
    $thrDir = Join-Path $OutRoot ("threshold_" + $r.leg)
    $t = Run-ThresholdSweep -Ckpt $ck -AnchorDir $r.anchor -ThrRoot $thrDir
    if ($t) {
        $thrResults += [pscustomobject]@{ src = $r.leg; min_f1 = $t.min_f1; mean_f1 = $t.mean_f1; note = "threshold_tuned" }
    }
}

# --- Promote best overall ---
$allCandidates = @()
foreach ($r in $longRows) {
    $allCandidates += [pscustomobject]@{ leg = $r.leg; min_f1 = [double]$r.min_f1; mean_f1 = [double]$r.mean_f1; source = "long" }
}
foreach ($r in $sweepRows) {
    $allCandidates += [pscustomobject]@{ leg = $r.leg; min_f1 = [double]$r.min_f1; mean_f1 = [double]$r.mean_f1; source = "sweep" }
}
foreach ($b in $baselineRows) {
    $allCandidates += [pscustomobject]@{ leg = $b.name; min_f1 = [double]$b.min_f1; mean_f1 = [double]$b.mean_f1; source = "baseline" }
}

$promoteDir = Join-Path $OutRoot "promoted"
$bestLeg = $null
$bestMin = -1.0
$bestMean = 0.0
foreach ($c in $allCandidates) {
    if ($c.min_f1 -gt $bestMin -or ($c.min_f1 -eq $bestMin -and $c.mean_f1 -gt $bestMean)) {
        $bestMin = $c.min_f1
        $bestMean = $c.mean_f1
        $bestLeg = $c.leg
    }
}

if ($bestLeg -and $bestLeg -ne "ladder_promoted" -and $bestLeg -ne "adapt_recovery") {
    $srcDir = if ($bestLeg -like "fi*") {
        Join-Path $OutRoot ("sweep_ladder_m6/" + $bestLeg)
    } elseif (Test-Path (Join-Path $OutRoot ($bestLeg + "/clot_phi_best.pth"))) {
        Join-Path $OutRoot $bestLeg
    } else {
        $null
    }
    if ($srcDir -and (Test-Path (Join-Path $srcDir "clot_phi_best.pth"))) {
        New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
        Copy-Item (Join-Path $srcDir "clot_phi_best.pth") (Join-Path $promoteDir "clot_phi_best.pth") -Force
        $ma = Join-Path $srcDir "multi_anchor.jsonl"
        if (Test-Path $ma) {
            Copy-Item $ma (Join-Path $promoteDir "multi_anchor.jsonl") -Force
        }
    }
} elseif (Test-Path (Join-Path $RepoRoot $LadderPromotedCkpt)) {
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    Copy-Item (Join-Path $RepoRoot $LadderPromotedCkpt) (Join-Path $promoteDir "clot_phi_best.pth") -Force
    $bestLeg = "ladder_promoted_fallback"
    $lp = Join-Path $RepoRoot "outputs/biochem/gt_flow_ladder_6h/promoted/multi_anchor.jsonl"
    if (Test-Path $lp) { Copy-Item $lp (Join-Path $promoteDir "multi_anchor.jsonl") -Force }
    $bs = Summarize-MultiAnchor $lp
    if ($bs) { $bestMin = $bs.min_f1; $bestMean = $bs.mean_f1 }
}

Write-R2Log "promote" "OK" @{
    leg = $bestLeg
    min_f1 = $bestMin
    mean_f1 = $bestMean
    gate = $MinF1Gate
    beat_gate = ($bestMin -ge $MinF1Gate)
}

@{
    baselines = $baselineRows
    sweep = $sweepRows
    sweep_winner = @{ fi = $bestFi; mat = $bestMat }
    long = $longRows
    threshold = $thrResults
    promoted_leg = $bestLeg
    promoted_min_f1 = $bestMin
    promoted_mean_f1 = $bestMean
    min_f1_gate = $MinF1Gate
    beat_gate = ($bestMin -ge $MinF1Gate)
} | ConvertTo-Json -Depth 6 | Set-Content $SummaryPath -Encoding utf8

Write-Host "[OK] round2 complete -> $SummaryPath" -ForegroundColor Green

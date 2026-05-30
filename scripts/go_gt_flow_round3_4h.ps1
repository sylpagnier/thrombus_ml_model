# GT-flow round 3 (~4h): push min_f1 toward 0.38 from round2 promoted (adapt cache winner).
# No m8 dump. Optional 14ep clot-band teacher + adapt re-dump (min-steps 4).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_round3_4h.ps1"
#   ... -SkipTeacher -SkipFinetune

param(
    [switch] $SkipTeacher,
    [switch] $SkipFinetune,
    [switch] $SkipThreshold,
    [int] $TeacherEpochs = 14,
    [int] $FinetuneEpochs = 90,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 4,
    [double] $MinF1Gate = 0.38,
    [double] $Fi = 2.0,
    [double] $Mat = 2.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_gt_flow_env.ps1")
. (Join-Path $PSScriptRoot "_gt_flow_round_helpers.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_round3_4h"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "round3_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

$AdaptCache = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt"
$Round2Promoted = "outputs/biochem/gt_flow_round2_4h/promoted/clot_phi_best.pth"
$Round2Blend = "outputs/biochem/gt_flow_round2_4h/long_adapt_blend/clot_phi_best.pth"
$TeacherInit = "outputs/biochem/biochem_teacher_last.pth"

$initCkpt = $Round2Promoted
if (-not (Test-Path (Join-Path $RepoRoot $initCkpt))) {
    $initCkpt = $Round2Blend
}
if (-not (Test-Path (Join-Path $RepoRoot $initCkpt))) {
    Write-GtFlowLog -LogPath $LogPath -Step "preflight" -Status "FAIL" @{ note = "missing round2 ckpt" }
    exit 1
}
Write-GtFlowLog -LogPath $LogPath -Step "preflight" -Status "OK" @{ init = $initCkpt }

$legRows = @()
$thrResults = @()

# --- Threshold on round2 promoted (no retrain) ---
if (-not $SkipThreshold) {
    $t0 = Invoke-GtFlowThresholdSweep -LogPath $LogPath `
        -Ckpt (Join-Path $RepoRoot $initCkpt) -AnchorDir $AdaptCache `
        -ThrRoot (Join-Path $OutRoot "threshold_round2_promoted")
    if ($t0) {
        $thrResults += [pscustomobject]@{ src = "round2_promoted_thr"; min_f1 = $t0.min_f1; mean_f1 = $t0.mean_f1 }
        $legRows += [pscustomobject]@{
            leg = "round2_promoted_thr"
            anchor = $AdaptCache
            min_f1 = $t0.min_f1
            mean_f1 = $t0.mean_f1
        }
    }
}

# --- Optional teacher refresh + adapt re-dump (min-steps 4, not m8) ---
$trainAnchor = $AdaptCache
if (-not $SkipTeacher) {
    if (Test-Path (Join-Path $RepoRoot $TeacherInit)) {
        Write-GtFlowLog -LogPath $LogPath -Step "teacher_round3" -Status "START" @{ epochs = $TeacherEpochs }
        Copy-Item (Join-Path $RepoRoot $TeacherInit) (Join-Path $RepoRoot "outputs/biochem/biochem_teacher_best_high_mu.pth") -Force
        Set-PassiveGtFlowEnv -RunNote "round3_teacher_cb" -ClotBandMask -AdrBackprop:$false
        $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
        $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
        $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
        $env:BIOCHEM_EPOCHS = "$TeacherEpochs"
        $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
        $trc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
            --epochs $TeacherEpochs --save-best --run-name round3_teacher_cb
        if ($trc -eq 0) {
            $gateRc = Invoke-PythonRc scripts/check_passive_teacher_gate.py --run-note round3_teacher_cb -Quiet
            Write-GtFlowLog -LogPath $LogPath -Step "teacher_round3" -Status $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ gate = $gateRc }
            $trainAnchor = Join-Path $OutRoot "anchors_round3_stride36_m4"
            $drc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
                --teacher outputs/biochem/biochem_teacher_last.pth `
                --out-dir $trainAnchor --device cuda --time-stride $DumpStride --min-steps $DumpMinSteps --force
            if ($drc -eq 0) {
                Write-GtFlowLog -LogPath $LogPath -Step "dump_round3" -Status "OK" @{ out = $trainAnchor }
            } else {
                Write-GtFlowLog -LogPath $LogPath -Step "dump_round3" -Status "FAIL" @{ exit = $drc }
                $trainAnchor = $AdaptCache
            }
        } else {
            Write-GtFlowLog -LogPath $LogPath -Step "teacher_round3" -Status "FAIL" @{ exit = $trc }
        }
    }
}

# --- Finetune legs from round2 promoted ---
if (-not $SkipFinetune) {
    $finetuneSpecs = @(
        @{
            step = "finetune_adapt_blend"
            dir = "finetune_adapt_blend"
            anchor = $AdaptCache
            pred = "1"
            alpha = "0.75"
            fi = $Fi
            mat = $Mat
            lr = "5e-4"
        },
        @{
            step = "finetune_adapt_fi30"
            dir = "finetune_adapt_fi30"
            anchor = $AdaptCache
            pred = "1"
            alpha = "0.75"
            fi = 3.0
            mat = $Mat
            lr = "5e-4"
        },
        @{
            step = "finetune_newdump_blend"
            dir = "finetune_newdump_blend"
            anchor = $trainAnchor
            pred = "1"
            alpha = "0.75"
            fi = $Fi
            mat = $Mat
            lr = "5e-4"
        }
    )
    foreach ($spec in $finetuneSpecs) {
        if (-not (Test-Path (Join-Path $RepoRoot $spec.anchor))) {
            Write-GtFlowLog -LogPath $LogPath -Step $spec.step -Status "WARN" @{ note = "missing anchor" }
            continue
        }
        if ($spec.step -eq "finetune_newdump_blend" -and $spec.anchor -eq $AdaptCache) {
            continue
        }
        $legDir = Join-Path $OutRoot $spec.dir
        $s = Invoke-GtFlowClotLeg -LogPath $LogPath -StepName $spec.step -AnchorDir $spec.anchor -LegDir $legDir `
            -Epochs $FinetuneEpochs -Fi $spec.fi -Mat $spec.mat -PredSpecies $spec.pred -Alpha $spec.alpha `
            -Lr $spec.lr -InitCkpt $initCkpt
        if ($s) {
            @{ anchor = $spec.anchor } | ConvertTo-Json | Set-Content (Join-Path $legDir "leg_meta.json")
            $legRows += [pscustomobject]@{
                leg = $spec.dir
                anchor = $spec.anchor
                min_f1 = $s.min_f1
                mean_f1 = $s.mean_f1
                mean_logMAE = $s.mean_logMAE
            }
        }
    }
}

# --- Threshold top finetune ckpts ---
$sorted = $legRows | Sort-Object -Property @{ Expression = { [double]$_.min_f1 }; Descending = $true } | Select-Object -First 2
foreach ($r in $sorted) {
    $ck = Join-Path $OutRoot ($r.leg + "/clot_phi_best.pth")
    if (-not (Test-Path $ck)) { continue }
    $thrDir = Join-Path $OutRoot ("threshold_" + $r.leg)
    $t = Invoke-GtFlowThresholdSweep -LogPath $LogPath -Ckpt $ck -AnchorDir $r.anchor -ThrRoot $thrDir
    if ($t) {
        $thrResults += [pscustomobject]@{ src = $r.leg; min_f1 = $t.min_f1; mean_f1 = $t.mean_f1; note = "threshold_tuned" }
        $idx = ($legRows | ForEach-Object { $_.leg }).IndexOf($r.leg)
        if ($idx -ge 0) {
            $legRows[$idx].min_f1 = $t.min_f1
            $legRows[$idx].mean_f1 = $t.mean_f1
        }
    }
}

$candidates = @()
foreach ($r in $legRows) {
    $candidates += [pscustomobject]@{ leg = $r.leg; min_f1 = [double]$r.min_f1; mean_f1 = [double]$r.mean_f1 }
}
# baseline round2 eval on adapt (for promote comparison)
$baseEv = Join-Path $OutRoot "baseline_round2_promoted.jsonl"
$env:CLOT_PHI_ANCHOR_DIR = $AdaptCache
$brc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint (Join-Path $RepoRoot $initCkpt) --out $baseEv -Quiet
if ($brc -eq 0) {
    $bs = Summarize-MultiAnchor $baseEv
    if ($bs) {
        Write-GtFlowLog -LogPath $LogPath -Step "baseline_round2_promoted" -Status "OK" @{
            min_f1 = $bs.min_f1
            mean_f1 = $bs.mean_f1
        }
        $candidates += [pscustomobject]@{ leg = "round2_baseline"; min_f1 = $bs.min_f1; mean_f1 = $bs.mean_f1 }
    }
}

$promo = Invoke-GtFlowPromote -LogPath $LogPath -OutRoot $OutRoot -Candidates $candidates `
    -MinF1Gate $MinF1Gate -FallbackCkpt (Join-Path $RepoRoot $initCkpt) -FallbackLeg "round2_promoted"

@{
    init_ckpt = $initCkpt
    train_anchor = $trainAnchor
    legs = $legRows
    threshold = $thrResults
    promoted_leg = $promo.leg
    promoted_min_f1 = $promo.min_f1
    promoted_mean_f1 = $promo.mean_f1
    min_f1_gate = $MinF1Gate
    beat_gate = $promo.beat_gate
    recipe = @{ fi = $Fi; mat = $Mat; finetune_epochs = $FinetuneEpochs; teacher_epochs = $TeacherEpochs }
} | ConvertTo-Json -Depth 6 | Set-Content $SummaryPath -Encoding utf8

Write-Host "[OK] round3 complete -> $SummaryPath" -ForegroundColor Green

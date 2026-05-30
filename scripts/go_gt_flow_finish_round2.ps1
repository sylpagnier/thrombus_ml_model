# Finish gt_flow_round2_4h after long legs completed (threshold + promote; optional teacher tail).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gt_flow_finish_round2.ps1"
#   ... -RunTeacherTail   # only if round2 died during teacher_refresh

param(
    [switch] $RunTeacherTail,
    [int] $TeacherRefreshEpochs = 10,
    [int] $LongEpochs = 65,
    [double] $MinF1Gate = 0.34
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_gt_flow_env.ps1")
. (Join-Path $PSScriptRoot "_gt_flow_round_helpers.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\gt_flow_round2_4h"
$LogPath = Join-Path $OutRoot "round2_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"
$LadderM6 = "outputs/biochem/gt_flow_ladder_6h/anchors_stride36_m6"
$TeacherInit = "outputs/biochem/biochem_teacher_last.pth"
$bestFi = 2.0
$bestMat = 2.0

$longRows = Get-GtFlowLongRowsFromDisk -OutRoot $OutRoot
Write-Host "[i] loaded $($longRows.Count) long/finetune legs from disk" -ForegroundColor Cyan

if ($RunTeacherTail) {
    if (Test-Path (Join-Path $RepoRoot $TeacherInit)) {
        Write-GtFlowLog -LogPath $LogPath -Step "teacher_refresh" -Status "START" @{ epochs = $TeacherRefreshEpochs }
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
                Write-GtFlowLog -LogPath $LogPath -Step "teacher_refresh_dump" -Status "OK" @{ out = $refreshAnchor }
                $legDir = Join-Path $OutRoot "long_refresh_blend"
                $s = Invoke-GtFlowClotLeg -LogPath $LogPath -StepName "long_refresh_blend" -AnchorDir $refreshAnchor `
                    -LegDir $legDir -Epochs $LongEpochs -Fi $bestFi -Mat $bestMat -PredSpecies "1" -Alpha "0.75"
                if ($s) {
                    @{ anchor = $refreshAnchor } | ConvertTo-Json | Set-Content (Join-Path $legDir "leg_meta.json")
                    $longRows += [pscustomobject]@{
                        leg = "long_refresh_blend"
                        anchor = $refreshAnchor
                        min_f1 = $s.min_f1
                        mean_f1 = $s.mean_f1
                        mean_logMAE = $s.mean_logMAE
                    }
                }
            } else {
                Write-GtFlowLog -LogPath $LogPath -Step "teacher_refresh_dump" -Status "FAIL" @{ exit = $drc }
            }
        } else {
            Write-GtFlowLog -LogPath $LogPath -Step "teacher_refresh" -Status "FAIL" @{ exit = $rc }
        }
    }
}

$thrResults = @()
$sorted = $longRows | Sort-Object -Property @{ Expression = { [double]$_.min_f1 }; Descending = $true } | Select-Object -First 3
foreach ($r in $sorted) {
    $anchor = $r.anchor
    if (-not $anchor) {
        if ($r.leg -like "*adapt*") {
            $anchor = "outputs/biochem/passive_species_clotband_focus/anchors_clotband_adapt"
        } else {
            $anchor = $LadderM6
        }
    }
    $ck = Join-Path $OutRoot ($r.leg + "/clot_phi_best.pth")
    if (-not (Test-Path $ck)) { continue }
    $thrDir = Join-Path $OutRoot ("threshold_" + $r.leg)
    $t = Invoke-GtFlowThresholdSweep -LogPath $LogPath -Ckpt $ck -AnchorDir $anchor -ThrRoot $thrDir
    if ($t) {
        $thrResults += [pscustomobject]@{ src = $r.leg; min_f1 = $t.min_f1; mean_f1 = $t.mean_f1; note = "threshold_tuned" }
        $idx = @($longRows | ForEach-Object { $_.leg }).IndexOf($r.leg)
        if ($idx -ge 0) {
            $longRows[$idx].min_f1 = $t.min_f1
            $longRows[$idx].mean_f1 = $t.mean_f1
        }
    }
}

$candidates = @()
foreach ($r in $longRows) {
    $candidates += [pscustomobject]@{ leg = $r.leg; min_f1 = [double]$r.min_f1; mean_f1 = [double]$r.mean_f1 }
}
foreach ($t in $thrResults) {
    $candidates += [pscustomobject]@{ leg = ($t.src + "_thr"); min_f1 = [double]$t.min_f1; mean_f1 = [double]$t.mean_f1 }
}

$fallback = Join-Path $RepoRoot "outputs/biochem/gt_flow_round2_4h/long_adapt_blend/clot_phi_best.pth"
$promo = Invoke-GtFlowPromote -LogPath $LogPath -OutRoot $OutRoot -Candidates $candidates `
    -MinF1Gate $MinF1Gate -FallbackCkpt $fallback -FallbackLeg "long_adapt_blend"

@{
    long = $longRows
    threshold = $thrResults
    promoted_leg = $promo.leg
    promoted_min_f1 = $promo.min_f1
    promoted_mean_f1 = $promo.mean_f1
    min_f1_gate = $MinF1Gate
    beat_gate = $promo.beat_gate
} | ConvertTo-Json -Depth 6 | Set-Content $SummaryPath -Encoding utf8

Write-Host "[OK] finish round2 -> $SummaryPath" -ForegroundColor Green

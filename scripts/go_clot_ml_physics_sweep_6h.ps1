# Physics-guided clot ML architecture sweep (pred GINO-DEQ, 6 anchors).
# Default ~6h (9 legs @ 40ep). Use -TargetHours 1 for ~1h resume of remaining legs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_physics_sweep_6h.ps1"
#   powershell ... -Legs step2_d30,step3_onset -TargetHours 1
#   powershell ... -Legs step1_a55,step7b_a55 -DryRun
#
# Morning:
#   python scripts/summarize_clot_ml_physics_sweep_6h.py

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $MixtureCkpt = "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [string] $Val = "patient007",
    [int] $Epochs = 40,
    [double] $TargetHours = 0,
    [string[]] $Legs = @(),
    [switch] $SkipViz,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_ML_DEVICE = "cuda"
$env:PYTHONUNBUFFERED = "1"
# Per-leg timeline PNG (full rule_mixture rollout + GINO-DEQ); add to budget below.
$VizMinutesPerLeg = 20

$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_clot_ml_physics_6h"
$ManifestPath = Join-Path $SweepDir "sweep_log.jsonl"
New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null

# Physics hypotheses (wall shear / stagnation / progressive commit / per-vessel onset)
$LegCatalog = [ordered]@{
    ref_rule_mixture = @{
        Minutes = 4
        Physics = "baseline"
        Hypothesis = "Frozen expert-mix shell (neg70+base); reference viz only"
        Train = $null
        VizPivot = "rule_mixture"
    }
    step1_a35 = @{
        Minutes = 42
        Physics = "residual_on_inc40"
        Hypothesis = "Local BCE residual on hand-tuned inc40 progressive top-k shell"
        Train = @("scripts/train_clot_ml_step1_residual.py", "--alpha", "0.35")
        CkptName = "clot_ml_step1_best.pth"
        VizFlag = "--step1-ckpt"
    }
    step1_a55 = @{
        Minutes = 42
        Physics = "residual_on_inc40"
        Hypothesis = "Stronger residual (alpha=0.55) on inc40 shell"
        Train = @("scripts/train_clot_ml_step1_residual.py", "--alpha", "0.55")
        CkptName = "clot_ml_step1_best.pth"
        VizFlag = "--step1-ckpt"
    }
    step7b_a35 = @{
        Minutes = 90
        Physics = "residual_on_expert_mix"
        Hypothesis = "Residual on frozen rule_mixture shell (alpha=0.35)"
        Train = @("scripts/train_clot_ml_step7b_hybrid.py", "--alpha", "0.35", "--mixture-ckpt", $MixtureCkpt)
        CkptName = "clot_ml_step7b_best.pth"
        VizFlag = "--step7b-ckpt"
    }
    step7b_a55 = @{
        Minutes = 90
        Physics = "residual_on_expert_mix"
        Hypothesis = "Stronger residual on rule_mixture shell (alpha=0.55)"
        Train = @("scripts/train_clot_ml_step7b_hybrid.py", "--alpha", "0.55", "--mixture-ckpt", $MixtureCkpt)
        CkptName = "clot_ml_step7b_best.pth"
        VizFlag = "--step7b-ckpt"
    }
    step2_d30 = @{
        Minutes = 75
        Physics = "seed_rank_gnn"
        Hypothesis = "GNN re-ranks wall-half risk before hard top-k (delta=0.30)"
        Train = @("scripts/train_clot_ml_step2_band_gnn.py", "--delta-scale", "0.30")
        CkptName = "clot_ml_step2_best.pth"
        VizFlag = "--step2-ckpt"
    }
    step2_d50 = @{
        Minutes = 75
        Physics = "seed_rank_gnn"
        Hypothesis = "Stronger GNN seed reorder (delta=0.50)"
        Train = @("scripts/train_clot_ml_step2_band_gnn.py", "--delta-scale", "0.50")
        CkptName = "clot_ml_step2_best.pth"
        VizFlag = "--step2-ckpt"
    }
    step3_onset = @{
        Minutes = 75
        Physics = "per_vessel_onset"
        Hypothesis = "Learned onset gate (stagnation timing) on inc40 pool"
        Train = @("scripts/train_clot_ml_step3_temporal_gate.py")
        CkptName = "clot_ml_step3_best.pth"
        VizFlag = "--step3-ckpt"
    }
    pivot_mixture = @{
        Minutes = 38
        Physics = "expert_mixture"
        Hypothesis = "Retrain softmax over shear-rule families (neg70/base/...)"
        Train = @("scripts/train_clot_ml_pivot.py", "--pivot", "rule_mixture")
        CkptName = "clot_ml_pivot_rule_mixture_best.pth"
        VizFlag = "--pivot-ckpt"
    }
}

$DefaultOrder = @(
    "ref_rule_mixture",
    "step1_a35",
    "step1_a55",
    "step7b_a35",
    "step7b_a55",
    "step2_d30",
    "step2_d50",
    "step3_onset",
    "pivot_mixture"
)

if ($Legs.Count -eq 1 -and ($Legs[0] -match ',')) {
    $Legs = @($Legs[0] -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$RunOrder = if ($Legs.Count -gt 0) { @($Legs) } else { $DefaultOrder }

# Scale epochs when a wall-clock budget is set (uses leg catalog minutes @ 40ep).
$trainLegIds = @(
    $RunOrder | Where-Object {
        $LegCatalog.Contains($_) -and $null -ne $LegCatalog[$_].Train
    }
)
if ($TargetHours -gt 0 -and $trainLegIds.Count -gt 0) {
    $estTrainMinAt40 = 0
    foreach ($lid in $trainLegIds) { $estTrainMinAt40 += [int]$LegCatalog[$lid].Minutes }
    $vizMin = if ($SkipViz) { 0 } else { $VizMinutesPerLeg * $trainLegIds.Count }
    $targetMin = $TargetHours * 60.0
    $trainBudget = $targetMin - $vizMin
    if ($trainBudget -lt 15.0) {
        Write-Host "[WARN] TargetHours=$TargetHours too tight for $($trainLegIds.Count) legs + viz (~${vizMin}min); use -SkipViz or fewer legs" -ForegroundColor Yellow
        $trainBudget = [Math]::Max(15.0, $targetMin * 0.85)
    }
    $scaled = [int][Math]::Round(40.0 * $trainBudget / [double]$estTrainMinAt40)
    $Epochs = [Math]::Max(6, [Math]::Min(40, $scaled))
}

function Write-SweepLog {
    param([string] $Leg, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        leg_id = $Leg
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    ($row | ConvertTo-Json -Compress) + "`n" | Add-Content -Path $ManifestPath -Encoding utf8
}

function Save-LegMeta {
    param(
        [string] $LegDir,
        [string] $LegId,
        [hashtable] $Leg,
        [string] $CkptRel,
        [string] $PngRel,
        [string] $Status
    )
    $meta = [ordered]@{
        leg_id = $LegId
        physics = $Leg.Physics
        hypothesis = $Leg.Hypothesis
        epochs = $Epochs
        ckpt = $CkptRel
        viz_png = $PngRel
        status = $Status
    }
    $meta | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $LegDir "leg_meta.json") -Encoding utf8
}

Write-Host ""
$budgetTag = if ($TargetHours -gt 0) { "~${TargetHours}h budget" } else { "~6h target" }
Write-Host "[NEW] clot ML physics sweep ($budgetTag, pred GINO-DEQ)" -ForegroundColor Cyan
Write-Host "[i] sweep_dir=$SweepDir legs=$($RunOrder -join ',') epochs=$Epochs" -ForegroundColor DarkGray
if ($TargetHours -gt 0 -and $trainLegIds.Count -gt 0) {
    Write-Host "[i] TargetHours=$TargetHours scaled epochs for $($trainLegIds.Count) train legs" -ForegroundColor DarkGray
}

$totalMin = 0
foreach ($lid in $RunOrder) {
    if (-not $LegCatalog.Contains($lid)) { continue }
    $base = [int]$LegCatalog[$lid].Minutes
    if ($null -ne $LegCatalog[$lid].Train) {
        $totalMin += [int][Math]::Round($base * $Epochs / 40.0)
        if (-not $SkipViz) { $totalMin += $VizMinutesPerLeg }
    }
    else {
        $totalMin += $base
    }
}
Write-Host "[i] estimated_minutes=$totalMin (~$([math]::Round($totalMin / 60.0, 1)) h) [6-anchor eval legs are slower than step1]" -ForegroundColor DarkGray

foreach ($legId in $RunOrder) {
    if (-not $LegCatalog.Contains($legId)) {
        Write-Host "[WARN] unknown leg=$legId (skip)" -ForegroundColor Yellow
        continue
    }
    $leg = $LegCatalog[$legId]
    $legDir = Join-Path $SweepDir $legId
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    $pngRel = "outputs/biochem/sweep_clot_ml_physics_6h/$legId/temporal_rule_${Anchor}_timeline.png"
    $pngPath = Join-Path $RepoRoot $pngRel

    Write-Host ""
    Write-Host "[NEW] leg=$legId physics=$($leg.Physics)" -ForegroundColor Cyan
    Write-Host "[i] $($leg.Hypothesis)" -ForegroundColor DarkGray

    if ($DryRun) {
        Write-SweepLog -Leg $legId -Status "dry_run"
        continue
    }

    $ckptRel = ""
    try {
        if ($legId -eq "ref_rule_mixture") {
            $ckptRel = $MixtureCkpt
            if (-not $SkipViz) {
                Invoke-PythonRcCheck -Label "$legId viz" -PyArgs @(
                    "scripts/viz_clot_temporal_rule_timeline.py",
                    "--anchor", $Anchor,
                    "--anchor-dir", $AnchorDir,
                    "--keyframes", "8",
                    "--pivot-ckpt", $MixtureCkpt,
                    "--vel-source", "kinematics",
                    "--kine-ckpt", $KineCkpt,
                    "--out", $pngRel
                )
            }
            $pivotSummary = Join-Path $RepoRoot "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/summary.json"
            if (Test-Path $pivotSummary) {
                Copy-Item -Force $pivotSummary (Join-Path $legDir "summary.json")
            }
            Save-LegMeta -LegDir $legDir -LegId $legId -Leg $leg -CkptRel $ckptRel -PngRel $pngRel -Status "ref_only"
            Write-SweepLog -Leg $legId -Status "ok" -Data @{ ckpt = $ckptRel; png = $pngRel }
            continue
        }

        $trainArgs = [System.Collections.Generic.List[string]]::new()
        foreach ($t in $leg.Train) { $trainArgs.Add([string]$t) }
        $trainArgs.Add("--anchor-dir"); $trainArgs.Add($AnchorDir)
        $trainArgs.Add("--step0-json"); $trainArgs.Add($Step0Json)
        $trainArgs.Add("--val"); $trainArgs.Add($Val)
        $trainArgs.Add("--epochs"); $trainArgs.Add("$Epochs")
        $trainArgs.Add("--out-dir"); $trainArgs.Add("outputs/biochem/sweep_clot_ml_physics_6h/$legId")

        Invoke-PythonRcCheck -Label "$legId train" -PyArgs $trainArgs.ToArray()

        $ckptRel = "outputs/biochem/sweep_clot_ml_physics_6h/$legId/$($leg.CkptName)"

        if (-not $SkipViz) {
            Write-Host "[i] leg=$legId viz (silent ~15-30min: rule_mixture rollout + GINO-DEQ + PNG)..." -ForegroundColor DarkGray
            $vizArgs = @(
                "scripts/viz_clot_temporal_rule_timeline.py",
                "--anchor", $Anchor,
                "--anchor-dir", $AnchorDir,
                "--keyframes", "8",
                $leg.VizFlag, $ckptRel,
                "--vel-source", "kinematics",
                "--kine-ckpt", $KineCkpt,
                "--out", $pngRel
            )
            Invoke-PythonRcCheck -Label "$legId viz" -PyArgs $vizArgs
        }

        Save-LegMeta -LegDir $legDir -LegId $legId -Leg $leg -CkptRel $ckptRel -PngRel $pngRel -Status "ok"
        Write-SweepLog -Leg $legId -Status "ok" -Data @{ ckpt = $ckptRel; png = $pngRel }
        Write-Host "[OK] leg=$legId ckpt=$ckptRel" -ForegroundColor Green
    }
    catch {
        Save-LegMeta -LegDir $legDir -LegId $legId -Leg $leg -CkptRel $ckptRel -PngRel $pngRel -Status "fail"
        Write-SweepLog -Leg $legId -Status "fail" -Data @{ error = $_.Exception.Message }
        Write-Host "[ERR] leg=$legId $($_.Exception.Message)" -ForegroundColor Red
        throw
    }
}

Write-Host ""
Invoke-PythonRcCheck -Label "sweep summarize" -PyArgs @(
    "scripts/summarize_clot_ml_physics_sweep_6h.py",
    "--sweep-dir", "outputs/biochem/sweep_clot_ml_physics_6h"
)

Write-Host ""
Write-Host "[OK] sweep complete -> $SweepDir\sweep_summary.json" -ForegroundColor Green
Write-Host "[i] Compare PNGs under outputs/biochem/sweep_clot_ml_physics_6h/*/temporal_rule_${Anchor}_timeline.png"

param(
    [string] $RunRoot = "outputs/biochem/eda/autonomous_6h",
    [switch] $SkipPhase1,
    [switch] $SkipPhase2
)

# Autonomous ~6h GPU block (docs/WALL_MODEL_PLAN.md s10-s11).
#
# Chosen for expected value given everything measured this session:
#
# PHASE 1 (~2h, NO training) -- monetize the s10.4 finding at cohort scale.
#   s10.4 found 34% of vessels have the flow->clot relation INVERTED, and that a single
#   deployable t=0 statistic (band_speed_q25) tells the regimes apart at 93.8% / 90.6% LOO
#   (recalibrated to 0.0822 for predicted flow -- probe_regime_route.py). s2.7 had concluded
#   a global gate percentile was "the ceiling of what a flow-only post-process can do"
#   precisely BECAUSE you could not tell which vessels it harms. Phase 1 measures the actual
#   F1 payoff of routing across every clot-rich vessel: gate OFF vs global vs regime-routed.
#   This is deploy-legal, needs no retraining, and directly tests whether routing converts
#   s2.7's minimax tradeoff into a net win.
#
# PHASE 2 (~4h, training) -- the prerequisite for ALL further training work (s11.3 change B).
#   v1-v4 each failed for reasons unrelated to the knob being tuned. s9.12/s9.14 found the
#   common cause: the objective supervises 5-10 step TBPTT windows while deploy is a 200-step
#   free rollout, so loss moves 0.2% while deploy F1 swings 0.37->0.61, the FP term never
#   fires, and the rolled-state brake moves the rollout ~1%. The only loss term that sees a
#   rolled-out state is the deploy_horizon aux -- capped at 40 of ~200 steps, i.e. blind to
#   where over-painting accumulates. v5 raises it to 150; v3 (already run) is the control.
#   The question is NOT "does v5 score higher" -- it is "does decreasing loss now TRACK
#   deploy score", which is what makes every future training experiment interpretable.
#
#   .\scripts\go_autonomous_6h.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "run.log"
$startAll = Get-Date

function Say($msg, $color = "Cyan") {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $Log -Value $line
}

Say "=== AUTONOMOUS 6h BLOCK START ===" "Green"
Say "out=$OutDir"

# ---------------------------------------------------------------- PHASE 1
if (-not $SkipPhase1) {
    Say "PHASE 1: full-cohort regime-routed gate sweep (no training)"
    # Clot-rich vessels from the s10 physical EDA, minus the sealed one-shot holdouts.
    # patient043 IS included: s9.15 already spent it (0.6497 banked) and this is a
    # read-only re-grade of the same rollout, not a new tuning decision.
    # Regime-BALANCED subset, not all 32: measured throughput is ~10 min/vessel, so the
    # full cohort would eat ~5h of the 6h budget and starve phase 2. 6 inverted + 6 normal
    # spanning the band_speed_q25 range, including the s2.7 vessels routing is meant to fix.
    # (patient021/037/035/032/020/043 are covered by the separately-running 6-vessel sweep.)
    $anchors = @(
        "patient019","patient025","patient029","patient011","patient024","patient001",
        "patient002","patient006","patient010","patient013","patient040","patient041"
    ) -join ","
    $p1out = Join-Path $OutDir "phase1_regime_gate_cohort.json"
    Say "  12 regime-balanced anchors (6 inverted / 6 normal), one rollout each, 3 gradings per rollout"
    $rc = Invoke-PythonRc "scripts/diag_regime_gate_sweep.py" `
        "--anchors" $anchors "--pct" "25" "--out" $p1out
    if ($rc -ne 0) { Say "PHASE 1 FAILED (rc=$rc) -- continuing to phase 2" "Yellow" }
    else { Say "PHASE 1 done -> $p1out" "Green" }
} else { Say "PHASE 1 skipped" "DarkGray" }

$afterP1 = Get-Date
Say ("elapsed after phase 1: {0:hh\:mm\:ss}" -f ($afterP1 - $startAll))

# ---------------------------------------------------------------- PHASE 2
if (-not $SkipPhase2) {
    Say "PHASE 2: v5 (full-horizon deploy aux, 40 -> 150) vs v3 control"
    Say "  v3 already ran; v5 is the single-variable arm. Watch whether loss now tracks deploy F1."
    $rc = Invoke-PythonRc "-m" "src.training.train_species_pushforward_continuous" `
        "--phase" "biochem_gnn" "--recipe" "mat_growth_simple" `
        "--leg" "WG_stenosis_subcohort_ft_v5" `
        "--out" (Join-Path $OutDir "WG_stenosis_subcohort_ft_v5/best.pth") `
        "--epochs" "12" "--early-stop" "6" "--lr" "5e-5" `
        "--anchors" "patient039,patient040,patient041,patient042,patient044" `
        "--val-anchor" "patient043" "--exclude-val-from-train" `
        "--drop-xy" "--deploy-freq" "1" `
        "--init" "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth" `
        "--init-mode" "full"
    if ($rc -ne 0) { Say "PHASE 2 training rc=$rc (may be mass-reject / OOM -- check log)" "Yellow" }
    else { Say "PHASE 2 training done" "Green" }
} else { Say "PHASE 2 skipped" "DarkGray" }

Say ("=== DONE. total elapsed {0:hh\:mm\:ss} ===" -f ((Get-Date) - $startAll)) "Green"
Say "Phase 1 payoff:  $OutDir/phase1_regime_gate_cohort.json"
Say "Phase 2 log:     $OutDir/WG_stenosis_subcohort_ft_v5/train_log.jsonl"
Say "KEY QUESTION for phase 2: does loss now correlate with deploy_clot_f1 across epochs?"
Say "  (v3 baseline: loss 61.36-61.47 flat while F1 swung 0.366-0.613 -- no correlation)"

# =============================================================================
# One-shot wall-model sweep: multi-hop flow features (6 GPU-h budget)
#
# Root cause (docs/WALL_MODEL_PLAN.md s2): 97% of patient020's false positives are a
# DISTANT wrong pocket (median 56 hops), not adjacent overpaint. The graded label lives
# on wall nodes where u=v=0 by no-slip, so the model's 1-hop flow feature is chance-level
# at locating clot (LOVO AUC 0.512). hop1+hop2 neighbourhood speed lifts that to 0.643
# (oracle-sign 0.741), and all three honest holdouts sit in the regime where it is
# strongest (p020 0.950, p043 0.998, p044 0.718).
#
# Stages write JSON before the next begins, so a partial run is still informative.
# Every stage is skipped if its output already exists -- safe to re-run after a crash.
#
#   powershell -ExecutionPolicy Bypass -File scripts/go_wall_multihop_sweep.ps1
# =============================================================================
param(
    [int]    $TrainEpochs   = 6,
    [double] $Lr            = 2e-5,
    [string] $Holdout       = "patient020",
    [string] $ChallengeSet  = "patient043,patient044",
    [switch] $SkipChallenge,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/eda/multihop_sweep"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$Log = Join-Path $OutRoot "sweep_log.txt"
$SweepStart = Get-Date

function Say([string]$m) {
    $ts = (Get-Date).ToString("HH:mm:ss")
    $el = [int]((Get-Date) - $SweepStart).TotalMinutes
    $line = "[$ts +${el}m] $m"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $Log -Value $line
}

# Deploy-faithful eval. NOTE: closed-loop coupling is deliberately left OFF -- the local
# corrector is directionally inverted (speeds flow up on clot nodes where GT slows it
# -41.7%) and its clot mask is ~97.6% phantom. See plan s7.1-7.2.
Remove-Item Env:SPECIES_CLOSED_LOOP_COUPLING -ErrorAction SilentlyContinue

Say "=== STAGE A: free probes (CPU) ==="

$probeJson = "outputs/biochem/eda/probe_multihop.json"
if ($Force -or -not (Test-Path $probeJson)) {
    $rc = Invoke-PythonRc scripts/probe_multihop_flow.py
    if ($rc -ne 0) { Say "probe FAILED rc=$rc"; exit 1 }
} else { Say "probe cached -> $probeJson" }

$probe = Get-Content $probeJson -Raw | ConvertFrom-Json
Say ("probe verdict: " + $probe.verdict)
if ($probe.verdict -like "STOP*") {
    Say "GATE FAILED: probe says the feature carries no held-out signal. Stopping before GPU spend."
    exit 2
}

# Burden predictability, now including flow features (s3 tested geometry only).
if ($Force -or -not (Test-Path "outputs/biochem/eda/burden_predictability_clotrich.json")) {
    Invoke-PythonRc scripts/eda_burden_predictability.py --min-burden 0.005 --n-perm 2000 `
        --out outputs/biochem/eda/burden_predictability_clotrich.json | Out-Null
}

Say "=== STAGE B: train 2 arms (GPU) ==="
# WG_multihop      : hop1+hop2 flow block ON
# WG_multihop_ctrl : identical stack + warm start, feature OFF
# The control is what makes any gain attributable to the feature rather than to the
# warm start / lr / epoch budget. Do not drop it to save time.
#
# Cohort is the N+ checkpoint's OWN 12 training anchors, passed explicitly. Do NOT use
# --all-anchors: that would pull patient020 into training and destroy the holdout.
$NplusAnchors = "patient001,patient005,patient006,patient007,patient010,patient013," +
                "patient016,patient021,patient029,patient032,patient035,patient037"
$InitCkpt = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth"

$arms = @("WG_multihop", "WG_multihop_ctrl")
foreach ($arm in $arms) {
    $armOut = "$OutRoot/$arm"
    $ckpt = "$armOut/best.pth"
    if (-not $Force -and (Test-Path $ckpt)) { Say "$arm cached"; continue }
    Say "training $arm (epochs=$TrainEpochs lr=$Lr)"
    # lr 2e-5, not the historical 1e-4: all four wide-cohort arms peaked at epoch 1 and
    # degraded monotonically at 1e-4 -- a divergence signature (plan s4).
    $rc = Invoke-PythonRc `
        -m src.training.train_species_pushforward_continuous `
        --phase biochem_gnn `
        --recipe mat_growth_simple `
        --leg $arm `
        --init $InitCkpt `
        --init-mode full `
        --anchors $NplusAnchors `
        --val-anchor $Holdout `
        --exclude-val-from-train `
        --epochs $TrainEpochs `
        --lr $Lr `
        --early-stop 4 `
        --deploy-freq 2 `
        --out $ckpt
    if ($rc -ne 0) { Say "$arm training FAILED rc=$rc -- continuing to eval what exists" }
}

Say "=== STAGE C: held-out eval (GPU) ==="
foreach ($arm in $arms) {
    $ckpt = "$OutRoot/$arm/best.pth"
    if (-not (Test-Path $ckpt)) { Say "${arm}: no checkpoint, skipping eval"; continue }
    $evalOut = "$OutRoot/$arm/eval_$Holdout.json"
    if ($Force -or -not (Test-Path $evalOut)) {
        Say "eval $arm on $Holdout"
        Invoke-PythonRc scripts/eval_mat_growth_simple.py --ckpt $ckpt `
            --anchors $Holdout --no-baseline --cheap-val --out $evalOut | Out-Null
    }
    # FP geography is the metric that actually tests the hypothesis: did the DISTANT
    # wrong pocket shrink? F1 alone cannot distinguish that from a threshold shift.
    $geoOut = "$OutRoot/$arm/fpgeo_$Holdout.json"
    if ($Force -or -not (Test-Path $geoOut)) {
        Say "FP geography $arm on $Holdout"
        Invoke-PythonRc scripts/diag_fp_geography.py --ckpt $ckpt `
            --anchors $Holdout --out $geoOut | Out-Null
    }
}

if (-not $SkipChallenge) {
    Say "=== STAGE D: challenge pair (only for the winning arm) ==="
    $best = $null; $bestF1 = -1.0
    foreach ($arm in $arms) {
        $f = "$OutRoot/$arm/eval_$Holdout.json"
        if (-not (Test-Path $f)) { continue }
        $r = Get-Content $f -Raw | ConvertFrom-Json
        $f1 = [double]$r.simple.mean.deploy_clot_f1
        Say ("  {0}: deploy_clot_f1={1:N4} mass={2:N3}" -f $arm, $f1, [double]$r.simple.mean.deploy_clot_mass_ratio)
        if ($f1 -gt $bestF1) { $bestF1 = $f1; $best = $arm }
    }
    if ($best -and $bestF1 -gt 0.50) {
        Say "winner=$best (F1=$bestF1) beats the 0.500 baseline -> evaluating $ChallengeSet"
        Invoke-PythonRc scripts/eval_mat_growth_simple.py --ckpt "$OutRoot/$best/best.pth" `
            --anchors $ChallengeSet --no-baseline --cheap-val `
            --out "$OutRoot/$best/eval_challenge.json" | Out-Null
    } else {
        Say "no arm beat 0.500 (best=$bestF1) -- skipping challenge pair, it would not be informative"
    }
}

Say "=== STAGE E: report (free) ==="
Invoke-PythonRc scripts/summarize_multihop_sweep.py --root $OutRoot `
    --baseline-f1 0.500 --baseline-mass 2.418 --out "$OutRoot/SUMMARY.md"

Say "=== DONE -> $OutRoot/SUMMARY.md ==="
Get-Content "$OutRoot/SUMMARY.md" | Write-Host

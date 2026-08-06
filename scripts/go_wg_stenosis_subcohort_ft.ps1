param(
    [int]    $Epochs         = 15,
    [int]    $EarlyStop      = 6,
    [double] $Lr             = 5e-5,   # all three legs are frozen-backbone head FTs
    [string] $TrainAnchors   = "",
    [string] $HoldoutAnchor  = "patient043",
    [double] $GatePct        = 25,
    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_stenosis_subcohort",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth",
    [string] $Leg            = "WG_stenosis_subcohort_ft_v4",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit
)

# Stenosis/aneurysm sub-cohort recall fine-tune (docs/WALL_MODEL_PLAN.md s9).
# Zero-shot floor (NO training on this cohort): WG_clotrich_nplus + flow gate pct=25 already
# scores deploy_clot_f1=0.650 on patient043 (oracle ceiling 0.697 -- selection is ~93% there).
# The diagnosed gap is UNDER-seeding (mass_ratio=0.653, front_speed=0.862, FN=44 vs FP=11), the
# mirror image of the over-seeding problem the rest of the wall-gen ladder was built to fix.
#
# v1 (WG_stenosis_subcohort_ft) flipped underpred:fp 2.0:8.0 -> 4.0:4.0 in one step and
# REGRESSED: deploy_clot_f1 0.650 -> 0.522 by overshooting straight through balance into
# over-seeding (mass 0.653 -> 2.59, front_speed 0.862 -> 2.99, FP 11 -> 157). Kept in
# mat_growth_simple.py as the exact historical record of that run -- do not "fix" it in place.
#
# v2 (WG_stenosis_subcohort_ft_v2) fixed five diagnosed causes (s9.9): halved the loss-ratio
# move, added a symmetric mass guard, gated training-time selection to match the final eval,
# widened train-window coverage (train_t0_coverage_frac), and graded a sliding window instead
# of a single point. v2 ALSO regressed -- but usefully: its arrest probe (s9.10) found the
# model reaches a genuinely better state mid-rollout on patient043 (F1 0.732 at t=130, beating
# the 0.650 zero-shot floor AND every s9.4 oracle ceiling) and then overshoots it for another
# ~70 steps to F1 0.499 (mass 1.41 -> 2.75 WITHIN ONE EPOCH). Root cause: v3_config (what both
# v1 and v2 trained on) has NO differentiable signal opposing continued growth once GT clot is
# captured -- step/final mass+FP penalties were at 0.0 in both.
#
# The growth-arrest probe (s9.10, scripts/probe_growth_arrest.py, zero-shot warm-start across
# the s9.4 cohort) then found the real defect, and it is NOT "no arrest": the model's clot
# ONSET is anti-correlated with the truth, perfectly monotone in deep clot mass (n=5) --
#   deep  0 -> GT t=55, model t=18 (-37 EARLY)   039      deep 68 -> GT t=20, model t=80 (+60 LATE) 042
#   deep  8 -> GT t=60, model t=20 (-40 EARLY)   040      deep 74 -> GT t=20, model t=60 (+40 LATE) 041
#   deep  9 -> GT t=60, model t=20 (-40 EARLY)   043 (holdout)
# i.e. the vessels that clot early AND thick are exactly the ones it starts latest on. On the
# holdout the LOCATION is already right (precision 0.96 at t=80, 0.83 at t_final -- the 11
# nodes it fires early are all TP by t=80); the whole t_final deficit is FN=42 / recall 0.558 /
# mass 0.674. The holdout needs MORE growth, correctly timed -- not braking.
#
# v3 (WG_stenosis_subcohort_ft_v3) was EXACTLY v2 plus ONE mechanism -- a deliberate
# single-mechanism A/B (s9.11). IT RAN AND WAS A CLEAN NEGATIVE (s9.12): the brake moved the
# rollout ~1% (front_speed 4.545 -> 4.605, t_final mass 4.02 -> 4.03) on a model 400% off
# target, and every epoch was still mass-rejected. Its design was:
#   - Adds the GT-relative, time-resolved growth brake (step_mass_penalty 0.75 /
#     step_prec_fp_penalty 0.5 / final_mass_penalty 1.5 at target 1.2 / final_prec_fp_penalty
#     1.0 / mature_fp_exempt=False) -- WG_prec_iter's own validated values, not new code.
#     rolled_final_mass_fp_penalty is GT-relative at EVERY unroll step: while GT is still empty
#     n_gt clamps to 1, so a premature commit of N nodes gives mass_ratio=N and softplus(N-1.2)
#     fires hard. It is therefore a PREMATURE-FIRING suppressor (what 039/040/043 need), and
#     being GT-relative it stays SILENT on 041/042 where the model is behind GT. Correct on
#     both halves of a cohort that splits early/late.
#   - Keeps v2's recall pressure (underpred 3.0 / fp 6.0) and frozen backbone UNCHANGED. 4 of 5
#     vessels incl. the holdout are under-grown and the brake is silent below target, so there
#     was nothing to protect them from; and the holdout's location is already correct, so the
#     defect is rate/onset, which the readout heads govern. Holding both fixed is what makes
#     this an attributable test of the brake itself.
#   - Selection-only change (cannot confound the training A/B -- it just picks among epochs):
#     symmetric select_front_speed_target_lambda / select_fp_fn_imbalance_lambda replace two
#     confirmed-dead terms. The old select_front_speed_lambda rewards min(front_speed, 1.5), so
#     it saturated to a flat +0.30 every epoch (front_speed ran 2.5-5.06) AND rewarded overshoot
#     on the way there; the old select_fn_fp_lambda only fires FN-heavy, so it read 0.000 every
#     epoch once the regime turned FP-heavy.
#   - Keeps v2's other fixes: gated selection, t0 coverage, sliding-window eval + f1_min floor,
#     and select_mass_hard_max -- now anchored to t_final, not the sliding-window mean (s9.10:
#     mass only grows over this rollout, so the mean systematically understated the risk).
#
# v4 (WG_stenosis_subcohort_ft_v4, the default here) -- comparing all four legs against their
# OBSERVED t_final mass on patient043 finally isolates the driver, and it is none of the knobs
# v1/v2/v3 were tuning (s9.12):
#     leg                underpred   fp    mass          leg           underpred  fp   mass
#     WG_clotrich_nplus      2.0    16.0   0.674         v1               4.0    4.0  4.200
#     WG_prec_iter           1.0    16.0   1.109         v2               3.0    6.0  ~4.02
#                                                        v3 (+brake)      3.0    6.0  4.032
# underpred 4.0 -> 3.0 (a 33% cut) moves mass 4%: nearly inert. fp_weight splits the table
# perfectly -- every leg at 16.0 controls mass, every leg that blew up had it CUT to 4-6.
# v1 cut it and v2/v3 inherited the cut, because fp_weight is not set by the geom/flux stack
# these legs inherit (so it takes the recipe's 16.0) but was DOCUMENTED as PushforwardConfig's
# bare 8.0 default -- making "6.0" look like a mild reduction when it was a 2.7x cut.
# v4 is v3 with that ONE value restored: fp_weight 6.0 -> 16.0. Brake kept so its effect stays
# readable against v2. v3-vs-v4 is therefore a clean single-variable test of fp_weight.
#
# v5 (WG_stenosis_subcohort_ft_v5) = v3 + ONE change: deploy_horizon/aux_cap 40 -> 150, testing
# whether making the objective SEE the deploy horizon makes decreasing loss track deploy score.
# IT RAN AND CHANGED NOTHING -- bit-identical to v3 (s12.2). Census found why: the deploy_horizon
# aux gets its OWN opt.zero_grad/backward/step, so it is 1 optimizer step out of 757, and
# grad_clip=1.0 neuters its magnitude too. Its length and weight are both irrelevant by
# construction.
#
# v6 (WG_stenosis_subcohort_ft_v6) attacks the same problem from the other side (s12.3): instead
# of making ONE term horizon-aware, make EVERY term horizon-aware -- curriculum_unroll
# True -> False (the curriculum was pinning unroll to 5 through epoch 10 regardless of config),
# unroll 5 -> 25 on all ~756 main windows, deploy_horizon_all_packs -> True. ~5x v3's compute
# (~35 min/epoch, ~3.5h for 6 epochs); tbptt_tail=5 keeps the extra cost forward-only.
# ITS SUCCESS CRITERION IS NOT F1: read train_log.jsonl and ask whether
# Spearman(loss, deploy_clot_score) across epochs went NEGATIVE. v3 and v5 both sat at +0.314
# -- weakly POSITIVE, i.e. lower loss trended toward WORSE deploy score. If v6 also fails to
# flip it, stop reweighting this objective and go to s11.3 change D (explicit autocatalysis).
#
# WARNING -- default train set deliberately departs from the codebase's sealed
# WALL_GEN_BATCH_1B_* split (train=012/040/041/042, challenge=043+044, exclude=039):
#   - includes patient039 (excluded there: half-finished sim, T=92, thinnest signal probed)
#   - trains on patient044 (there: sealed challenge alongside 043)
# This spends one of the two sealed challenge points; patient043 is what stays sealed for
# both this sub-study and the original wall-gen plan once this leg trains. See s9 for the
# tradeoff. Pass -TrainAnchors to override (e.g. the sealed WALL_GEN_BATCH_1B_TRAIN list).
#
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft_v6 -Epochs 6 -EarlyStop 6 -Fresh   # live leg
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft_v5 -Fresh   # reproduce v5 (== v3)
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft_v3 -Fresh   # reproduce v3
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft_v2 -Fresh   # reproduce v2
#   .\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft -Fresh      # reproduce v1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$KnownLegs = @("WG_stenosis_subcohort_ft", "WG_stenosis_subcohort_ft_v2", "WG_stenosis_subcohort_ft_v3", "WG_stenosis_subcohort_ft_v4", "WG_stenosis_subcohort_ft_v5", "WG_stenosis_subcohort_ft_v6")
# Every leg from v2 on bakes CLOT_POCKET_GATE_PCT=25 into its env_overrides.
$GatedLegs = @("WG_stenosis_subcohort_ft_v2", "WG_stenosis_subcohort_ft_v3", "WG_stenosis_subcohort_ft_v4", "WG_stenosis_subcohort_ft_v5", "WG_stenosis_subcohort_ft_v6")
if ($KnownLegs -notcontains $Leg) {
    Write-Host "[ERR] This launcher is for $($KnownLegs -join ' / ') only (got $Leg)" -ForegroundColor Red
    exit 1
}
if ($GatedLegs -contains $Leg -and $GatePct -ne 25) {
    Write-Host "[ERR] $Leg bakes CLOT_POCKET_GATE_PCT=25 into training-time selection (env_overrides)." -ForegroundColor Red
    Write-Host "[ERR] -GatePct $GatePct would grade the final eval at a DIFFERENT percentile than" -ForegroundColor Red
    Write-Host "[ERR] selection used -- change the leg's env_overrides in mat_growth_simple.py too, not just this flag." -ForegroundColor Red
    exit 1
}

$OutDir = Join-Path $RepoRoot $RunRoot
$ArmDir = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
$ArmLog = Join-Path $ArmDir "train_log.jsonl"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

if ([string]::IsNullOrWhiteSpace($TrainAnchors)) {
    $trainCsv = (python -c @"
from src.biochem_gnn.mat_growth_simple import wall_gen_stenosis_subcohort_train_anchors
print(','.join(wall_gen_stenosis_subcohort_train_anchors(holdout='$HoldoutAnchor')))
"@).Trim()
    if (-not $trainCsv -or $LASTEXITCODE -ne 0) {
        Write-Host "[ERR] failed to resolve stenosis sub-cohort train anchors (holdout=$HoldoutAnchor)" -ForegroundColor Red
        exit 1
    }
} else {
    $trainCsv = $TrainAnchors
}
$trainList = @($trainCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$forbidden = @("patient002", "patient023", $HoldoutAnchor)
foreach ($bad in $forbidden) {
    if ($trainList -contains $bad) {
        Write-Host "[ERR] train list must not include $bad" -ForegroundColor Red
        Write-Host "[i] train=$trainCsv" -ForegroundColor DarkGray
        exit 1
    }
}
if ($trainList.Count -lt 2 -or $trainList.Count -gt 8) {
    Write-Host "[ERR] stenosis sub-cohort expects 2-8 train vessels, got $($trainList.Count)" -ForegroundColor Red
    exit 1
}
if ($trainList -contains "patient039") {
    Write-Host "[WARN] patient039 is in train -- excluded from WALL_GEN_CLOT_RICH_ANCHORS elsewhere" -ForegroundColor Yellow
    Write-Host "[WARN]   (half-finished sim, T=92; weakest signal of any vessel in the s9 probe)" -ForegroundColor Yellow
}
if ($trainList -contains "patient044") {
    Write-Host "[WARN] patient044 is in train -- sealed WALL_GEN_BATCH_1B_CHALLENGE alongside 043" -ForegroundColor Yellow
    Write-Host "[WARN]   patient043 remains the only vessel sealed for both this leg and the main plan" -ForegroundColor Yellow
}

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing N+ warm-start ckpt: $InitCkpt" -ForegroundColor Red
    Write-Host "[i] Run go_wg_clotrich_nplus.ps1 first, or point -InitCkpt elsewhere." -ForegroundColor DarkGray
    exit 1
}

Write-Host "[NEW] wg_stenosis_subcohort_ft ($Leg): $Epochs ep / ES $EarlyStop / lr=$Lr" -ForegroundColor Cyan
Write-Host "[i] goal=recall FT (FN down, front_speed up) on the stenosis/aneurysm sub-cohort" -ForegroundColor DarkGray
if ($Leg -eq "WG_stenosis_subcohort_ft_v6") {
    Write-Host "[i] v6 = s11.3 change B, retry 2: make EVERY loss term horizon-aware, not just the aux" -ForegroundColor DarkGray
    Write-Host "[i]   curriculum_unroll True->False, unroll 5->25 on all ~756 windows; deploy_horizon_all_packs->True" -ForegroundColor DarkGray
    Write-Host "[i]   why: v5 lengthened the aux 40->150 and was BIT-IDENTICAL to v3 -- the aux is 1 opt step of 757," -ForegroundColor DarkGray
    Write-Host "[i]   so its length/weight cannot matter. Coverage/depth is the only remaining lever." -ForegroundColor DarkGray
    Write-Host "[i] cost: ~18.9k evals/epoch vs v3's 3.8k (~5x, ~35 min/epoch); tbptt_tail=5 keeps it forward-only" -ForegroundColor DarkGray
    Write-Host "[i] SUCCESS CRITERION IS NOT F1: does Spearman(loss, deploy_clot_score) across epochs go NEGATIVE?" -ForegroundColor Yellow
    Write-Host "[i]   v3 and v5 both sat at +0.314 (lower loss -> WORSE deploy). If v6 stays positive, go to s11.3 change D." -ForegroundColor Yellow
} elseif ($Leg -eq "WG_stenosis_subcohort_ft_v5") {
    Write-Host "[i] v5 = v3 + ONE change: deploy_horizon/aux_cap 40 -> 150" -ForegroundColor DarkGray
    Write-Host "[i] WARNING: v5 RAN and was BIT-IDENTICAL to v3 (s12.2) -- the aux is 1 optimizer step of 757," -ForegroundColor Yellow
    Write-Host "[i]   so lengthening it changes nothing. Use -Leg WG_stenosis_subcohort_ft_v6." -ForegroundColor Yellow
} elseif ($Leg -eq "WG_stenosis_subcohort_ft_v4") {
    Write-Host "[i] v4 = EXACTLY v3 + ONE value: fp_weight 6.0 -> 16.0 (the warm-start/prec_iter baseline v1 silently cut)" -ForegroundColor DarkGray
    Write-Host "[i] why: fp_weight splits every leg by observed t_final mass on p043 --" -ForegroundColor DarkGray
    Write-Host "[i]   fp=16 -> mass 0.674 (warm-start) / 1.109 (prec_iter)   |   fp=4-6 -> mass 4.200 (v1) / ~4.02 (v2) / 4.032 (v3)" -ForegroundColor DarkGray
    Write-Host "[i]   meanwhile underpred 4.0->3.0 (a 33% cut) moved mass only 4% -- the knob v1/v2/v3 tuned is nearly inert" -ForegroundColor DarkGray
    Write-Host "[i] brake kept from v3 (unchanged) so its ~1% effect stays readable against v2" -ForegroundColor DarkGray
} elseif ($Leg -eq "WG_stenosis_subcohort_ft_v3") {
    Write-Host "[i] v3 = EXACTLY v2 + one mechanism (single-mechanism A/B): the GT-relative time-resolved growth brake" -ForegroundColor DarkGray
    Write-Host "[i] WARNING: v3 RAN and the brake moved the rollout ~1% (front 4.545->4.605) -- use -Leg WG_stenosis_subcohort_ft_v4" -ForegroundColor Yellow
    Write-Host "[i] brake: step_mass=0.75 step_fp=0.5 final_mass=1.5 (target 1.2) final_fp=1.0 mature_fp_exempt=False -- v2 had all at 0" -ForegroundColor DarkGray
    Write-Host "[i] unchanged from v2: underpred=3.0 fp=6.0, freeze-backbone, t0_coverage=0.85, gate=25, sliding eval @0.65/1.0" -ForegroundColor DarkGray
    Write-Host "[i] why it fires: brake is GT-relative, so a premature commit while GT is still empty gives mass_ratio=N -> softplus(N-1.2)" -ForegroundColor DarkGray
    Write-Host "[i]   -> suppresses PREMATURE firing (039/040/043 start 37-40 steps early); stays silent on 041/042 which start LATE" -ForegroundColor DarkGray
    Write-Host "[i] select adds symmetric front_speed_target/fp_fn_imbalance (selection only -- does not confound the brake A/B)" -ForegroundColor DarkGray
} elseif ($Leg -eq "WG_stenosis_subcohort_ft_v2") {
    Write-Host "[i] stack=N+ (featfix_03) + underpred=3.0 fp=6.0 (was 2.0/16.0, v1 overshot at 4.0/4.0); freeze-backbone" -ForegroundColor DarkGray
    Write-Host "[i] select=strict clot F1 (0.70) + soft clout score (0.30) + front_speed/FN-FP panel" -ForegroundColor DarkGray
    Write-Host "[i] v2 fixes: mass_hard_max guard, gated selection (pct=25), t0_coverage=0.85, sliding eval @0.65/1.0 + f1_min floor" -ForegroundColor DarkGray
    Write-Host "[i] WARNING: v2's own arrest probe found it doesn't stop growing (s9.10) -- use -Leg WG_stenosis_subcohort_ft_v3" -ForegroundColor Yellow
} else {
    Write-Host "[i] stack=N+ (featfix_03) + underpred=4.0 fp=4.0 (was 2.0/16.0); freeze-backbone" -ForegroundColor DarkGray
    Write-Host "[i] select=strict clot F1 (0.70) + soft clout score (0.30) + front_speed/FN-FP panel" -ForegroundColor DarkGray
    Write-Host "[i] WARNING: this is v1, the config that regressed to deploy_clot_f1=0.522 -- use -Leg WG_stenosis_subcohort_ft_v3" -ForegroundColor Yellow
}
Write-Host "[i] train($($trainList.Count))=$trainCsv" -ForegroundColor DarkGray
Write-Host "[i] holdout=$HoldoutAnchor (deploy-faithful cold gate, pocket-gate pct=$GatePct)" -ForegroundColor DarkGray
Write-Host "[i] out=$ArmDir" -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random (--NoInit)" -ForegroundColor DarkGray
} else {
    Write-Host "[i] init=$InitCkpt" -ForegroundColor DarkGray
}
Write-Host "[i] zero-shot floor to beat: deploy_clot_f1=0.650 on patient043 (N+ + gate, no cohort training)" -ForegroundColor DarkGray

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host "[skip] $Leg already completed (eval JSON exists); pass -Fresh to rerun" -ForegroundColor DarkGray
    exit 0
}

if (-not $EvalOnly) {
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", $Leg,
        "--out", $ArmCkpt,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--lr", "$Lr",
        "--anchors", $trainCsv,
        "--val-anchor", $HoldoutAnchor,
        "--exclude-val-from-train",
        "--drop-xy",
        "--deploy-freq", "1"
    )
    if ($NoInit) {
        $trainArgs += "--no-init"
    } else {
        $trainArgs += @("--init", $InitCkpt, "--init-mode", "full")
    }

    $null = Invoke-PythonRcCheck -Label "$Leg train" -PyArgs $trainArgs

    if (-not (Test-Path $ArmCkpt)) {
        Write-Host "[ERR] $Leg failed to produce checkpoint (all epochs mass-rejected?)" -ForegroundColor Red
        exit 1
    }
} elseif (-not (Test-Path $ArmCkpt)) {
    Write-Host "[ERR] $Leg missing ckpt for -EvalOnly" -ForegroundColor Red
    exit 1
}

$null = Invoke-PythonRcCheck -Label "$Leg cold eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $ArmCkpt,
    "--no-baseline",
    "--anchors", $HoldoutAnchor,
    "--pocket-gate-pct", "$GatePct",
    "--out", $ArmHold
)

if (-not $SkipViz) {
    $vizDir = Join-Path $RepoRoot "outputs/biochem/viz/mat_growth"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    $vizOut = Join-Path $vizDir "clot_ladder_stenosis_subcohort_$HoldoutAnchor.png"
    Write-Host "[NEW] ladder viz -> $vizOut" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", $Leg,
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] eval=$ArmHold" -ForegroundColor DarkGray
Write-Host "[i] compare deploy_clot_f1 against the zero-shot floor 0.650 before claiming a win" -ForegroundColor DarkGray

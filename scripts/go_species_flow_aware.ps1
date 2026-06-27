# Flow-aware species teacher: retrain the GraphSAGE with a CLOT-AWARE flow channel so the local
# kinematic corrector finally has ROI.
#
# Why: the baseline teacher input is `[z_kin, sdf]` -- a clot-blind latent. The `--gt-flow` gate
# proved feeding perfect velocity is a no-op (the model has no flow feature), and oracle-mu
# regresses (the only live channel, a re-solved z_kin, is OOD). This run adds explicit flow
# features (speed + shear + divergence + geometry) computed from the GT COMSOL velocity in
# training (clot-aware), so flow -> Mat localization is actually learned. At deploy the
# corrector-coupled flow (now in-distribution via local tiling) supplies those features.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_species_flow_aware.ps1
#   powershell ... -Smoke          # 8 ep quick sanity (not for a winner pick)
#   powershell ... -Epochs 90
#
# Note: the input dim changes (+5 flow channels), so the run is FRESH (the old sage first layer
# cannot be warm-started). After training, re-gate with:
#   python -m src.tools.compare_coupled_mat_rollout --graph data/processed/graphs_biochem_anchors/patient007.pt `
#     --species-ckpt outputs/biochem/biochem_gnn/flow_aware/sage/species/best.pth --gt-flow
# A real positive --gt-flow delta on the retrained teacher is the green light for the corrector.

param(
    [int] $Epochs = 75,
    [int] $EarlyStop = 24,
    [double] $Lr = 1.5e-4,
    [int] $FlowFeatsTime = -1,    # representative GT time for the (static) flow features; -1 = last
    [double] $LatentDropout = 0.0, # latent leash: prob of zeroing z_kin per step (try 0.5). 0 = off
    [switch] $DynamicFlow,        # Trap C: time-varying flow features (per-step GT velocity)
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if ($Smoke) {
    $Epochs = 8
    $EarlyStop = 4
    Write-Host "[i] smoke mode: $Epochs ep" -ForegroundColor DarkGray
}

# Variant runs land in separate dirs so they don't clobber each other.
$RunName = "flow_aware"
if ($LatentDropout -gt 0.0) { $RunName = "flow_aware_leashed" }
if ($DynamicFlow) { $RunName = "${RunName}_dynamic" }
$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/$RunName"
$SageCkpt = Join-Path $RunRoot "sage/species/best.pth"

# --- canonical arch_ab sage recipe (kept in sync with go_biochem_gnn_arch_ab.ps1) -----------
$env:SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL = "1"
$env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "1"
$env:SPECIES_CONTINUOUS_DUAL_HEAD = "1"
$env:SPECIES_CONTINUOUS_PHYSICS_READOUT = "0"
$env:SPECIES_KIN_PER_VESSEL_NORM = "1"
$env:SPECIES_CONTINUOUS_SATURATION_GATE = "1"
$env:SPECIES_CONTINUOUS_MATURE_FP_EXEMPT = "1"
$env:SPECIES_CONTINUOUS_MATURE_FRAC = "0.95"
$env:SPECIES_CONTINUOUS_SATURATION_SCALE = "80"
$env:SPECIES_CONTINUOUS_TEMPORAL_GATE = "1"
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN = "0.5"
$env:SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX = "1.5"
$env:SPECIES_CONTINUOUS_VEL_DECAY = "1"
$env:SPECIES_CONTINUOUS_TEACHER_NOISE = "0.015"
$env:SPECIES_CONTINUOUS_TEACHER_FP_FRAC = "0.06"
$env:SPECIES_CONTINUOUS_TEACHER_BLUR = "0.22"
$env:SPECIES_CONTINUOUS_TBPTT_TAIL = "8"
$env:SPECIES_CONTINUOUS_CURRICULUM_UNROLL = "1"
$env:SPECIES_CONTINUOUS_CLOSED_LOOP_INIT = "0.35"
$env:SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT = "0.35"
$env:SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND = "1"
$env:SPECIES_CONTINUOUS_SPEED_FP_WEIGHT = "4.0"
$env:SPECIES_PUSHFORWARD_UNROLL = "12"
$env:SPECIES_PUSHFORWARD_MAX_UNROLL = "200"
$env:SPECIES_CONTINUOUS_DEPLOY_HORIZON = "200"
$env:SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL = "1"
$env:SPECIES_CONTINUOUS_DEPLOY_EVAL_DUAL = "1"
$env:SPECIES_CONTINUOUS_DEPLOY_DUAL_FULL_W = "0.70"
$env:SPECIES_TRAIN_DEPLOY_EVAL_FLOW = "kinematics"
$env:SPECIES_TRAIN_VEL_SOURCE = "gt"
$env:SPECIES_DEPLOY_HORIZON_ALL_PACKS = "1"
$env:SPECIES_DEPLOY_HORIZON_AUX_CAP = "72"
$env:SPECIES_CONTINUOUS_SCORE_CLOUT_W = "0.92"
$env:SPECIES_CONTINUOUS_CLOUT_SCORE = "guiding"
$env:CLOT_GUIDE_RELAX_HOPS = "2"
$env:CLOT_GUIDE_F_BETA = "0.5"
$env:CLOT_GUIDE_IOU_W = "0.45"
$env:CLOT_GUIDE_F05_W = "0.55"
$env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
$env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
$env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
$env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
$env:SPECIES_VISCOSITY_CALIB = "1"

# --- NEW: clot-aware flow features (this is the whole point of the run) ----------------------
$env:SPECIES_FLOW_FEATS = "1"            # append [speed, shear, divergence, x_n, y_n] to band inputs
$env:SPECIES_FLOW_FEATS_SOURCE = "gt"    # TRAINING: features from the clot-aware GT COMSOL velocity
$env:SPECIES_FLOW_FEATS_TIME = "$FlowFeatsTime"
$env:SPECIES_FLOW_FEATS_ABLATE = "0"     # keep flow signal during training
# --- NEW: latent leash (break z_kin dominance so the model actually reads the flow) ----------
$env:SPECIES_LATENT_DROPOUT = "$LatentDropout"
# --- NEW: Trap C -- time-varying flow features (per-step GT velocity), persisted in meta --------
$env:SPECIES_FLOW_FEATS_DYNAMIC = if ($DynamicFlow) { "1" } else { "0" }

Write-Host "[NEW] flow-aware sage species ($Epochs ep, lr=$Lr, flow_feats=gt@t$FlowFeatsTime, latent_dropout=$LatentDropout, dynamic=$DynamicFlow, FRESH)" -ForegroundColor Cyan
$pyArgs = @(
    "-m", "src.training.train_biochem_gnn",
    "--step", "species",
    "--all-anchors",
    "--val-anchor", "patient007",
    "--epochs", "$Epochs",
    "--early-stop", "$EarlyStop",
    "--lr", "$Lr",
    "--arch", "sage",
    "--species-out", $SageCkpt,
    "--fresh"
)
Invoke-PythonRcCheck -Label "flow_aware sage train" -PyArgs $pyArgs

Write-Host "[OK] checkpoint: $SageCkpt" -ForegroundColor Green
$gate = if ($DynamicFlow) { "--gt-flow-dynamic" } else { "--gt-flow" }
Write-Host "[i] re-gate (upper bound): python -m src.tools.compare_coupled_mat_rollout --graph data/processed/graphs_biochem_anchors/patient007.pt --species-ckpt $SageCkpt $gate" -ForegroundColor DarkGray
Write-Host "[i] Trap C headroom: compare static teacher's --gt-flow F1 vs this dynamic teacher's --gt-flow-dynamic F1" -ForegroundColor DarkGray
Write-Host "[i] leash check (flow ablation): set SPECIES_FLOW_FEATS_ABLATE=1 then run the tool; a leashed model's baseline F1 should DROP" -ForegroundColor DarkGray

param(
    [int]    $Epochs         = 25,
    [int]    $EarlyStop      = 8,
    [double] $Lr             = 1e-4,
    [string] $TrainAnchors   = "patient005,patient006,patient010",
    [string] $HoldoutAnchor  = "patient020",
    [string] $RunRoot        = "",
    [string] $InitCkpt       = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth",
    [ValidateSet("WG_prec_physfp", "WG_prec_cloop")]
    [string] $Leg            = "WG_prec_physfp",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipViz,
    [switch] $NoInit,
    [switch] $FpGeoOnly,
    [switch] $SkipFpGeo
)

# One FT from WG_prec_iter floor (NOT seed/front):
#   WG_prec_physfp  - physical_fp_gating (distant FP precision)
#   WG_prec_cloop   - closed_loop_init=0.85 + tbptt=12 (adjacent overpaint / drift)
# Gate: primary deploy_clot_f1 on patient020; mass in [0.5,1.5] hard; FN hard max 80.
#   .\scripts\go_wg_prec_physfp.ps1 -Fresh
#   .\scripts\go_wg_prec_physfp.ps1 -Leg WG_prec_cloop -Fresh
#   .\scripts\go_wg_prec_physfp.ps1 -FpGeoOnly

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if (-not $RunRoot) {
    if ($Leg -eq "WG_prec_cloop") {
        $RunRoot = "outputs/biochem/eda/wall_gen_prec_cloop"
    } else {
        $RunRoot = "outputs/biochem/eda/wall_gen_prec_physfp"
    }
}

$OutDir = Join-Path $RepoRoot $RunRoot
$ArmDir = Join-Path $OutDir $Leg
$ArmCkpt = Join-Path $ArmDir "best.pth"
$ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
$ArmLog = Join-Path $ArmDir "train_log.jsonl"
New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null

$vizDir = Join-Path $RepoRoot "outputs/biochem/viz/mat_growth"
New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
$FpGeoPng = Join-Path $vizDir "fp_geography_$HoldoutAnchor.png"
$FpGeoJson = Join-Path $vizDir "fp_geography_$HoldoutAnchor.json"

Write-Host "[NEW] fp_geography diagnostic on $HoldoutAnchor (WG_prec_iter floor)" -ForegroundColor Cyan
if (-not $SkipFpGeo -or $FpGeoOnly -or -not (Test-Path $FpGeoJson)) {
    $null = Invoke-PythonRcCheck -Label "fp_geography" -PyArgs @(
        "scripts/viz_fp_geography.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $InitCkpt,
        "--out", $FpGeoPng
    )
} else {
    Write-Host "[skip] fp_geography (existing $FpGeoJson); pass without -SkipFpGeo to refresh" -ForegroundColor DarkGray
}

if ($FpGeoOnly) {
    Write-Host "[OK] FpGeoOnly done -> $FpGeoPng" -ForegroundColor Green
    exit 0
}

# If caller left default Leg=physfp, honor viz recommendation when JSON exists.
if ((Test-Path $FpGeoJson) -and $Leg -eq "WG_prec_physfp" -and -not $PSBoundParameters.ContainsKey("Leg")) {
    try {
        $geo = Get-Content $FpGeoJson -Raw | ConvertFrom-Json
        $rec = [string]$geo.fp_geography.recommend_leg
        if ($rec -eq "cloop") {
            $Leg = "WG_prec_cloop"
            $RunRoot = "outputs/biochem/eda/wall_gen_prec_cloop"
            $OutDir = Join-Path $RepoRoot $RunRoot
            $ArmDir = Join-Path $OutDir $Leg
            $ArmCkpt = Join-Path $ArmDir "best.pth"
            $ArmHold = Join-Path $ArmDir "eval_holdout_cold.json"
            $ArmLog = Join-Path $ArmDir "train_log.jsonl"
            New-Item -ItemType Directory -Force -Path $ArmDir | Out-Null
            Write-Host "[i] viz recommended cloop -> switching Leg=$Leg" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARN] could not read fp_geography recommendation; keeping $Leg" -ForegroundColor DarkYellow
    }
}

$trainList = @($TrainAnchors.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$forbidden = @("patient002", "patient023", $HoldoutAnchor)
foreach ($bad in $forbidden) {
    if ($trainList -contains $bad) {
        Write-Host "[ERR] train list must not include $bad" -ForegroundColor Red
        exit 1
    }
}
if ($trainList.Count -lt 2 -or $trainList.Count -gt 8) {
    Write-Host "[ERR] physfp/cloop expects 2-8 train vessels, got $($trainList.Count)" -ForegroundColor Red
    exit 1
}
$trainCsv = [string]::Join(",", $trainList)

$InitPath = Join-Path $RepoRoot $InitCkpt
if (-not $NoInit -and -not (Test-Path $InitPath)) {
    Write-Host "[ERR] Missing prec_iter warm-start ckpt: $InitCkpt" -ForegroundColor Red
    exit 1
}

Write-Host ("[NEW] {0}: {1} ep / ES {2} / lr={3}" -f $Leg, $Epochs, $EarlyStop, $Lr) -ForegroundColor Cyan
Write-Host "[i] gate=primary deploy_clot_f1; mass hard [0.5,1.5]; FN hard max 80; never promote score alone" -ForegroundColor DarkGray
Write-Host ("[i] stack=WG_prec_iter loss + {0} only (no seed/front)" -f $Leg.Replace("WG_prec_","")) -ForegroundColor DarkGray
Write-Host ("[i] train({0})={1} holdout={2}" -f $trainList.Count, $trainCsv, $HoldoutAnchor) -ForegroundColor DarkGray
if ($NoInit) {
    Write-Host "[i] init=random" -ForegroundColor DarkGray
} else {
    Write-Host ("[i] init={0}" -f $InitCkpt) -ForegroundColor DarkGray
}

if ($Fresh) {
    Remove-Item -Force $ArmCkpt, $ArmHold, $ArmLog -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $ArmDir "best.json") -ErrorAction SilentlyContinue
}

if ((Test-Path $ArmHold) -and -not $Fresh -and -not $EvalOnly) {
    Write-Host "[skip] $Leg already completed; pass -Fresh to rerun" -ForegroundColor DarkGray
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
        Write-Host "[ERR] $Leg failed to produce checkpoint (all epochs mass/FN-rejected?)" -ForegroundColor Red
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
    "--out", $ArmHold
)

if (-not $SkipViz) {
    $vizOut = Join-Path $vizDir ("clot_ladder_" + $Leg.ToLower() + "_$HoldoutAnchor.png")
    $null = Invoke-PythonRcCheck -Label "$Leg ladder" -PyArgs @(
        "scripts/viz_mat_growth_clot_ladder.py",
        "--anchor", $HoldoutAnchor,
        "--ckpt", $ArmCkpt,
        "--arm-label", ($Leg.Replace("WG_prec_", "")),
        "--leg", $Leg,
        "--flow", "kinematics",
        "--out", $vizOut
    )
}

Write-Host "[OK] $Leg complete" -ForegroundColor Green
Write-Host "[i] ckpt=$ArmCkpt" -ForegroundColor DarkGray
Write-Host "[i] eval=$ArmHold" -ForegroundColor DarkGray
Write-Host "[i] fp_geo=$FpGeoPng" -ForegroundColor DarkGray

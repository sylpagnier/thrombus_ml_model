param(

    [int]    $Epochs         = 30,

    [int]    $EarlyStop      = 8,

    [string] $TrainAnchors   = "patient005,patient006,patient010,patient023,patient002",

    [string] $ValAnchor      = "patient020",

    [string] $HoldoutAnchors = "patient020",

    [string] $RunRoot        = "outputs/biochem/eda/wall_gen_physgat",

    [string] $ArmFilter      = "WG_physgat_ctrl,WG_physgat_01",

    [string] $FeatfixRefCkpt = "outputs/biochem/eda/wall_gen_featfix/WG_featfix_03/best.pth",

    [switch] $Fresh,

    [switch] $EvalOnly,

    [switch] $Viz

)



# Physics-biased GAT A/B vs featfix_03 (geom+flux SAGE). Deploy-faithful holdout patient020.

# Random init (no warm-start). Stage-A PM-GAT + mesh normals/SDF; no COMSOL UV into GAT.

# .\scripts\go_wg_physgat_ab.ps1 -Epochs 30 -EarlyStop 8 -Fresh



$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:PYTHONUNBUFFERED = "1"



$OutDir = Join-Path $RepoRoot $RunRoot

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null



Write-Host "[NEW] wg_physgat_ab: $Epochs ep / ES $EarlyStop (random init)" -ForegroundColor Cyan

Write-Host "[i] arms: fair SAGE ctrl + physics_gat (soft prior_scale=0.05, identity edge_proj)" -ForegroundColor DarkGray

Write-Host "[i] arch=physics_gat on featfix_03 feature stack (geom+flux, drop-xy, auto+coupled)" -ForegroundColor DarkGray

Write-Host "[i] train=$TrainAnchors val=$ValAnchor holdout=$HoldoutAnchors" -ForegroundColor DarkGray

Write-Host "[i] success bar: cold p020 score clearly above featfix_03 (~0.329); target >=0.36" -ForegroundColor DarkGray

Write-Host "[i] init=random (--no-init); deploy-faithful UV only (no COMSOL data.y steal)" -ForegroundColor DarkGray



$armsStr = (python -m src.training.train_species_pushforward_continuous --list-legs WG_physgat) | Out-String

$armKeys = @($armsStr.Trim() -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("[") -and $_.StartsWith("WG_physgat_") })



if ($ArmFilter) {

    $filters = @($ArmFilter.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })

    $armKeys = @($armKeys | Where-Object {

        $filters -contains $_ -or $filters -contains $_.Replace("WG_physgat_", "")

    })

}



if ($armKeys.Count -lt 1) {

    Write-Host "[ERR] No WG_physgat_* arms selected." -ForegroundColor Red

    exit 1

}



Write-Host "[i] arms ($($armKeys.Count)): $($armKeys -join ', ')" -ForegroundColor Cyan



foreach ($armId in $armKeys) {

    $armDir = Join-Path $OutDir $armId

    $armCkpt = Join-Path $armDir "best.pth"

    $armHold = Join-Path $armDir "eval_holdout_cold.json"

    New-Item -ItemType Directory -Force -Path $armDir | Out-Null



    if ($Fresh) {

        Remove-Item -Force $armCkpt, $armHold -ErrorAction SilentlyContinue

        Remove-Item -Force (Join-Path $armDir "best.json"), (Join-Path $armDir "train_log.jsonl") -ErrorAction SilentlyContinue

    }



    if ((Test-Path $armHold) -and -not $Fresh -and -not $EvalOnly) {

        Write-Host "[skip] $armId already completed (eval JSON exists)" -ForegroundColor DarkGray

        continue

    }



    Write-Host ""

    Write-Host "====== Arm ${armId} ======" -ForegroundColor Cyan



    if (-not $EvalOnly) {

        $trainArgs = @(

            "-m", "src.training.train_species_pushforward_continuous",

            "--phase", "biochem_gnn",

            "--recipe", "mat_growth_simple",

            "--leg", $armId,

            "--out", $armCkpt,

            "--epochs", "$Epochs",

            "--early-stop", "$EarlyStop",

            "--anchors", $TrainAnchors,

            "--val-anchor", $ValAnchor,

            "--exclude-val-from-train",

            "--drop-xy",

            "--deploy-freq", "1",

            "--no-init"

        )



        $null = Invoke-PythonRcCheck -Label "Arm $armId train" -PyArgs $trainArgs



        if (-not (Test-Path $armCkpt)) {

            Write-Host "[WARN] Arm $armId failed to produce checkpoint" -ForegroundColor Yellow

            continue

        }

    } elseif (-not (Test-Path $armCkpt)) {

        Write-Host "[WARN] Arm $armId missing ckpt for -EvalOnly" -ForegroundColor Yellow

        continue

    }



    $null = Invoke-PythonRcCheck -Label "Arm $armId eval" -PyArgs @(

        "scripts/eval_mat_growth_simple.py",

        "--ckpt", $armCkpt,

        "--no-baseline",

        "--anchors", $HoldoutAnchors,

        "--out", $armHold

    )

}



Write-Host ""

Write-Host "[NEW] summarizing wg_physgat_ab" -ForegroundColor Cyan

$null = Invoke-PythonRcCheck -Label "aggregate_physgat" -PyArgs @(

    "scripts/aggregate_sweep_v3.py",

    "--sweep-dir", $RunRoot,

    "--out-csv", (Join-Path $OutDir "physgat_ab_results.csv")

)



$refPath = Join-Path $RepoRoot $FeatfixRefCkpt

if (Test-Path $refPath) {

    Write-Host "[i] featfix_03 ref ckpt present: $FeatfixRefCkpt" -ForegroundColor DarkGray

    Write-Host "[i] compare cold p020 score to featfix_03 holdout (~0.329)" -ForegroundColor DarkGray

} else {

    Write-Host "[WARN] featfix_03 ref ckpt missing: $FeatfixRefCkpt (compare manually)" -ForegroundColor Yellow

}



if ($Viz) {

    $physCkpt = Join-Path $OutDir "WG_physgat_01\best.pth"

    if ((Test-Path $physCkpt) -and (Test-Path $refPath)) {

        $vizOut = Join-Path $RepoRoot "outputs/biochem/viz/mat_growth"

        New-Item -ItemType Directory -Force -Path $vizOut | Out-Null

        Write-Host "[NEW] viz physgat vs featfix_03 on patient020" -ForegroundColor Cyan

        $null = Invoke-PythonRcCheck -Label "viz_physgat_compare" -PyArgs @(

            "scripts/viz_mat_growth_clot_compare_hop.py",

            "--ckpt-a", $refPath,

            "--label-a", "featfix_03",

            "--ckpt-b", $physCkpt,

            "--label-b", "physgat_01",

            "--anchor", "patient020",

            "--out", (Join-Path $vizOut "clot_compare_featfix03_vs_physgat01_patient020.png")

        )

    } else {

        Write-Host "[WARN] -Viz skipped: need both physgat and featfix_03 ckpts" -ForegroundColor Yellow

    }

}



Write-Host "[OK] wg_physgat_ab done" -ForegroundColor Green


# Shared helpers for go_gt_flow_round2_4h.ps1 / go_gt_flow_round3_4h.ps1 / go_gt_flow_finish_round2.ps1

function Write-GtFlowLog {
    param(
        [string] $LogPath,
        [string] $Step,
        [string] $Status,
        [hashtable] $Data = @{}
    )
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
        [string] $ThreshSi = "0.045",
        [string] $Lr = "1e-3",
        [string] $InitCkpt = ""
    )
    Clear-ClotPhiEnv
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
    $env:CLOT_PHI_EPOCHS = "$Epochs"
    $env:CLOT_PHI_LR = $Lr
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
    $env:CLOT_PHI_SWEEP_DIR = (Split-Path $OutDir -Parent)
    $env:CLOT_PHI_SWEEP_LEG = $LegName
    if ($InitCkpt) {
        $env:CLOT_PHI_INIT_CHECKPOINT = $InitCkpt
    } else {
        Remove-Item Env:\CLOT_PHI_INIT_CHECKPOINT -ErrorAction SilentlyContinue
    }
}

function Invoke-GtFlowClotLeg {
    param(
        [string] $LogPath,
        [string] $StepName,
        [string] $AnchorDir,
        [string] $LegDir,
        [int] $Epochs,
        [double] $Fi,
        [double] $Mat,
        [string] $PredSpecies,
        [string] $Alpha,
        [string] $Lr = "1e-3",
        [string] $InitCkpt = ""
    )
    New-Item -ItemType Directory -Force -Path $LegDir | Out-Null
    Write-GtFlowLog -LogPath $LogPath -Step $StepName -Status "START" @{
        anchor = $AnchorDir
        epochs = $Epochs
        fi = $Fi
        mat = $Mat
        lr = $Lr
    }
    Set-ClotPhiRecipe -AnchorDir $AnchorDir -LegName (Split-Path $LegDir -Leaf) -OutDir $LegDir `
        -Epochs $Epochs -Fi $Fi -Mat $Mat -PredSpecies $PredSpecies -Alpha $Alpha -Lr $Lr -InitCkpt $InitCkpt
    $rc = Invoke-PythonRc -m src.training.train_clot_phi_simple
    if ($rc -ne 0) {
        Write-GtFlowLog -LogPath $LogPath -Step $StepName -Status "FAIL" @{ exit = $rc; phase = "train" }
        return $null
    }
    $ckpt = Join-Path $LegDir "clot_phi_best.pth"
    $evalOut = Join-Path $LegDir "multi_anchor.jsonl"
    $erc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out $evalOut -Quiet
    if ($erc -ne 0) {
        Write-GtFlowLog -LogPath $LogPath -Step $StepName -Status "FAIL" @{ exit = $erc; phase = "eval" }
        return $null
    }
    $s = Summarize-MultiAnchor $evalOut
    if ($s) {
        Write-GtFlowLog -LogPath $LogPath -Step $StepName -Status "OK" @{
            mean_f1 = $s.mean_f1
            min_f1 = $s.min_f1
            mean_logMAE = $s.mean_logMAE
        }
    }
    return $s
}

function Invoke-GtFlowThresholdSweep {
    param(
        [string] $LogPath,
        [string] $Ckpt,
        [string] $AnchorDir,
        [string] $ThrRoot,
        [string[]] $ThreshList = @("0.035", "0.040", "0.045", "0.050", "0.055", "0.060")
    )
    New-Item -ItemType Directory -Force -Path $ThrRoot | Out-Null
    $best = $null
    foreach ($thr in $ThreshList) {
        $env:CLOT_PHI_THRESH_SI = $thr
        $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir
        $tag = "thr_" + ($thr -replace '\.', '')
        $out = Join-Path $ThrRoot ($tag + ".jsonl")
        $rc = Invoke-PythonRc scripts/eval_clot_phi_multi_anchor.py --checkpoint $Ckpt --out $out -Quiet
        if ($rc -ne 0) { continue }
        $s = Summarize-MultiAnchor $out
        if ($s) {
            Write-GtFlowLog -LogPath $LogPath -Step ("threshold_" + $tag) -Status "OK" @{
                min_f1 = $s.min_f1
                mean_f1 = $s.mean_f1
                thr = $thr
            }
            if ($null -eq $best -or $s.min_f1 -gt $best.min_f1) { $best = $s }
        }
    }
    return $best
}

function Get-GtFlowLongRowsFromDisk {
    param([string] $OutRoot)
    $rows = @()
    foreach ($dir in Get-ChildItem -Path $OutRoot -Directory -ErrorAction SilentlyContinue) {
        $name = $dir.Name
        if ($name -notmatch '^(long_|finetune_)') { continue }
        $ma = Join-Path $dir.FullName "multi_anchor.jsonl"
        if (-not (Test-Path $ma)) { continue }
        $s = Summarize-MultiAnchor $ma
        if (-not $s) { continue }
        $anchor = ""
        $meta = Join-Path $dir.FullName "leg_meta.json"
        if (Test-Path $meta) {
            $anchor = (Get-Content $meta | ConvertFrom-Json).anchor
        }
        $rows += [pscustomobject]@{
            leg = $name
            anchor = $anchor
            min_f1 = $s.min_f1
            mean_f1 = $s.mean_f1
            mean_logMAE = $s.mean_logMAE
        }
    }
    return $rows
}

function Invoke-GtFlowPromote {
    param(
        [string] $LogPath,
        [string] $OutRoot,
        [array] $Candidates,
        [double] $MinF1Gate = 0.34,
        [string] $FallbackCkpt = "",
        [string] $FallbackLeg = "fallback"
    )
    $promoteDir = Join-Path $OutRoot "promoted"
    $bestLeg = $null
    $bestMin = -1.0
    $bestMean = 0.0
    foreach ($c in $Candidates) {
        if ($c.min_f1 -gt $bestMin -or ($c.min_f1 -eq $bestMin -and $c.mean_f1 -gt $bestMean)) {
            $bestMin = [double]$c.min_f1
            $bestMean = [double]$c.mean_f1
            $bestLeg = $c.leg
        }
    }
    $srcDir = $null
    if ($bestLeg) {
        $cand = Join-Path $OutRoot $bestLeg
        if (Test-Path (Join-Path $cand "clot_phi_best.pth")) {
            $srcDir = $cand
        } elseif (Test-Path (Join-Path $OutRoot ("sweep_ladder_m6/" + $bestLeg + "/clot_phi_best.pth"))) {
            $srcDir = Join-Path $OutRoot ("sweep_ladder_m6/" + $bestLeg)
        }
    }
    if (-not $srcDir -and $FallbackCkpt -and (Test-Path $FallbackCkpt)) {
        New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
        Copy-Item $FallbackCkpt (Join-Path $promoteDir "clot_phi_best.pth") -Force
        $bestLeg = $FallbackLeg
    } elseif ($srcDir) {
        New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
        Copy-Item (Join-Path $srcDir "clot_phi_best.pth") (Join-Path $promoteDir "clot_phi_best.pth") -Force
        $ma = Join-Path $srcDir "multi_anchor.jsonl"
        if (Test-Path $ma) {
            Copy-Item $ma (Join-Path $promoteDir "multi_anchor.jsonl") -Force
        }
    }
    Write-GtFlowLog -LogPath $LogPath -Step "promote" -Status "OK" @{
        leg = $bestLeg
        min_f1 = $bestMin
        mean_f1 = $bestMean
        gate = $MinF1Gate
        beat_gate = ($bestMin -ge $MinF1Gate)
    }
    return [pscustomobject]@{
        leg = $bestLeg
        min_f1 = $bestMin
        mean_f1 = $bestMean
        beat_gate = ($bestMin -ge $MinF1Gate)
    }
}

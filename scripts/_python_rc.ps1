# Reliable python exit codes under PowerShell $ErrorActionPreference=Stop.
# Dot-source: . (Join-Path $PSScriptRoot "_python_rc.ps1")
#
# When assigning exit code ($rc = Invoke-PythonRc ...), python stdout must not
# land in the return value (only the int exit code is returned).

function Invoke-PythonRc {
    param(
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs,
        [switch] $Quiet
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $myPid = $PID
    try {
        if ($Quiet) {
            & python -u @PyArgs 2>&1 | Out-Null
        } else {
            # Write-Host keeps console output off the success stream (so $rc = ... gets only exit code).
            # tqdm \r bars may render as one line per refresh vs direct python stdout.
            & python -u @PyArgs 2>&1 | ForEach-Object { Write-Host $_ }
        }
    } finally {
        # Cleanup any orphaned python.exe that is a child of this powershell process
        Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $myPid -and $_.Name -match "python" } | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    $rc = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
    $ErrorActionPreference = $prevEap
    return , $rc
}

function Invoke-PythonRcCheck {
    param(
        [Parameter(ValueFromRemainingArguments = $true)][string[]] $PyArgs,
        [string] $Label = "python"
    )
    $rc = Invoke-PythonRc @PyArgs
    if ($rc -ne 0) {
        Write-Host "[ERR] $Label failed (exit=$rc)" -ForegroundColor Red
        exit $rc
    }
    return 0
}

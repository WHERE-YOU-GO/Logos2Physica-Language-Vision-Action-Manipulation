$ErrorActionPreference = "Stop"

function Write-Step {
    param(
        [string]$Message
    )
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Get-Python311Command {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $version = & py -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.11") {
                return @("py", "-3.11")
            }
        } catch {
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        try {
            $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.11") {
                return @("python")
            }
        } catch {
        }
    }

    throw "Python 3.11 was not found. Install Python 3.11 and rerun this script."
}

function Invoke-Python {
    param(
        [string[]]$PythonCommand,
        [string[]]$Arguments
    )

    if ($PythonCommand.Length -gt 1) {
        $baseArgs = @($PythonCommand[1..($PythonCommand.Length - 1)])
        & $PythonCommand[0] @baseArgs @Arguments
    } else {
        & $PythonCommand[0] @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($PythonCommand -join ' ') $($Arguments -join ' ')"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Step "Detecting Python 3.11"
$pythonCommand = Get-Python311Command
Write-Host "Using Python command: $($pythonCommand -join ' ')"

$venvPath = Join-Path $repoRoot ".venv_win"

if (Test-Path $venvPath) {
    Write-Step "Reusing existing .venv_win"
} else {
    Write-Step "Creating .venv_win"
    Invoke-Python -PythonCommand $pythonCommand -Arguments @("-m", "venv", ".venv_win")
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed. Expected interpreter not found: $venvPython"
}

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in .venv_win."
}

Write-Step "Installing layered requirements"
$requirementFiles = @(
    "requirements/base.txt",
    "requirements/dev.txt",
    "requirements/demo.txt",
    "requirements/windows.txt"
)
foreach ($requirementFile in $requirementFiles) {
    Write-Host "Installing $requirementFile"
    & $venvPython -m pip install -r $requirementFile
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $requirementFile"
    }
}

Write-Step "Setting thread environment variables for this session"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
Write-Host "OPENBLAS_NUM_THREADS=1"
Write-Host "OMP_NUM_THREADS=1"
Write-Host "MKL_NUM_THREADS=1"
Write-Host "NUMEXPR_NUM_THREADS=1"

Write-Step "Windows-Dev environment is ready"
Write-Host "Activate:"
Write-Host ".\.venv_win\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Recommended commands:"
Write-Host "python -m pytest tests -vv"
Write-Host "python -m scripts.run_fsm_once --use_fake_robot --scene_dir data/scenes/scene_01 --backend demo"

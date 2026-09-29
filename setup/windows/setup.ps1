# One-time team setup for Windows (PowerShell).
#
# Easiest: double-click setup\windows\setup.bat
# Or from PowerShell at the repo root:
#   powershell -ExecutionPolicy Bypass -File setup\windows\setup.ps1
#   powershell -ExecutionPolicy Bypass -File setup\windows\setup.ps1 -NoHub
#
# What it does:
#   1. Creates .venv\ at the repo root (Python 3.12+)
#   2. Installs requirements.txt and the local `parkinson` package
#   3. Installs the lab skills for Bob into .bob\skills\
#   4. Signs you in to Skore Hub and writes .skore (gitignored)

param(
    [switch]$NoHub
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Invoke-Checked {
    # Run a native command and stop if it fails.
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

# --- 1. Python + virtual environment -----------------------------------------
Step "Finding Python 3.12+"
$Py = $null
$PyArgs = @()
$candidates = @(
    @{ Exe = "py"; Args = @("-3") },
    @{ Exe = "python"; Args = @() },
    @{ Exe = "python3"; Args = @() }
)
foreach ($c in $candidates) {
    if (Get-Command $c.Exe -ErrorAction SilentlyContinue) {
        try {
            & $c.Exe @($c.Args + @("-c", "import sys; sys.exit(sys.version_info < (3, 12))")) 2>$null
        } catch {
            continue
        }
        if ($LASTEXITCODE -eq 0) {
            $Py = $c.Exe
            $PyArgs = $c.Args
            break
        }
    }
}
if (-not $Py) {
    Write-Host "Python 3.12 or newer not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
    exit 1
}
& $Py @($PyArgs + @("--version"))

$VPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Step "Creating .venv\"
    Invoke-Checked $Py ($PyArgs + @("-m", "venv", ".venv"))
} else {
    Step ".venv\ already exists - reusing it"
}

# --- 2. Packages ---------------------------------------------------------------
Step "Installing requirements"
Invoke-Checked $VPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VPy @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $VPy @("-m", "pip", "install", "-e", ".")

# --- 3. Bob skills --------------------------------------------------------------
Step "Installing lab skills for Bob (.bob\skills\)"
$Skore = Join-Path $RepoRoot ".venv\Scripts\skore.exe"
Invoke-Checked $Skore @("skills", "install", "all", "--repo", "probabl-ai/skills-hackathon", "--agent", "bob")

# --- 4. Skore Hub ------------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "data") | Out-Null
if ($NoHub) {
    Step "Skipping Skore Hub sign-in (-NoHub)"
} elseif (Test-Path (Join-Path $RepoRoot ".skore")) {
    Step ".skore already exists - skipping Hub sign-in"
} else {
    Step "Signing in to Skore Hub (a browser window will open)"
    Write-Host "You must already be a member of the team workspace 'ibmhackathongroup1'."
    Invoke-Checked $VPy @("scripts\skore-agent")
}

# --- Check ---------------------------------------------------------------------
Step "Checking the environment"
$env:PYTHONIOENCODING = "utf-8"  # env_check prints unicode check marks
Invoke-Checked $VPy @("scripts\env_check.py")

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  - Activate the environment:  .venv\Scripts\Activate.ps1"
Write-Host "    (if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)"
Write-Host "  - Download the competition CSVs from the Kaggle Data tab into data\"

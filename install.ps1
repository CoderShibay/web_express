# Install web_express into its own folder, without touching the system Python packages.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Source SOURCE]
#
# SOURCE is what pip should install: a folder containing this project (the
# default is the folder this script is in), or a git address such as
# git+https://example.org/you/web_express.git
param([string]$Source = $PSScriptRoot)
$ErrorActionPreference = "Stop"

$HomeDir = if ($env:WEB_EXPRESS_HOME) { $env:WEB_EXPRESS_HOME } else { Join-Path $HOME ".web_express" }
$App = Join-Path $HomeDir "app"

$Python = $null
foreach ($candidate in @(@("py", "-3"), @("python"), @("python3"))) {
    $exe = $candidate[0]
    $rest = @($candidate | Select-Object -Skip 1)
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        & $exe @rest -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $Python = $candidate; break }
    }
}
if (-not $Python) {
    Write-Error "web_express needs Python 3.11 or newer. Install it from https://www.python.org/downloads/ and run this again."
    exit 1
}

Write-Host "Installing web_express into $App"
New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
$exe = $Python[0]
$rest = @($Python | Select-Object -Skip 1)
& $exe @rest -m venv $App
if ($LASTEXITCODE -ne 0) { exit 1 }
$Py = Join-Path $App "Scripts\python.exe"
$Wx = Join-Path $App "Scripts\web-express.exe"
& $Py -m pip install --quiet --upgrade pip
& $Py -m pip install --quiet $Source
if ($LASTEXITCODE -ne 0) { exit 1 }

& $Wx setup
if ($LASTEXITCODE -ne 0) { Write-Host "The ranking model will be downloaded on first search instead." }

Write-Host ""
Write-Host "web_express is installed: $Wx"
Write-Host ""
Write-Host "Next, register it with your AI agents:"
Write-Host "    $Wx init --dry-run     (shows what would change)"
Write-Host "    $Wx init"

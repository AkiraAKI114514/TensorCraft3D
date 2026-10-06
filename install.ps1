param([switch]$Training)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Install Node.js 20+ from https://nodejs.org first.' }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Install Python 3.10+ first.' }
npm install --registry=https://registry.npmjs.org
if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
if (-not (Test-Path .venv/Scripts/python.exe)) { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$requirements = if ($Training) { 'backend/requirements-training.txt' } else { 'backend/requirements.txt' }
& ./.venv/Scripts/python.exe -m pip install -r $requirements --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
Write-Host 'Ready. Run ./start.ps1'

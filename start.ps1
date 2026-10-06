$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed. Run install.ps1 first.' }
if (Test-Path .venv/Scripts/python.exe) { & ./.venv/Scripts/python.exe run.py }
else { python run.py }

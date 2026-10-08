param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed. Run install.ps1 first.' }
$arguments = @('run.py', '--port', $Port)
if ($NoBrowser) { $arguments += '--no-browser' }
if (Test-Path .venv/Scripts/python.exe) { & ./.venv/Scripts/python.exe @arguments }
else { python @arguments }

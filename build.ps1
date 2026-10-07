$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
if (-not (Test-Path release)) { New-Item -ItemType Directory -Path release | Out-Null }
$files = @('dist', 'backend', 'run.py', 'start.ps1', 'start.cmd', 'install.ps1', 'setup-training.ps1', 'build.ps1', 'README.md', 'package.json', 'package-lock.json', 'src', 'e2e', 'tests', 'playwright.config.ts', 'index.html', 'tsconfig.json', 'vite.config.ts', 'public')
Compress-Archive -Path $files -DestinationPath release/TensorCraft3D.zip -Force
Write-Host 'Built release/TensorCraft3D.zip'

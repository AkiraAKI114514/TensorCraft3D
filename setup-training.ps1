[CmdletBinding()]
param(
  [ValidateSet('cu128', 'cpu')]
  [string]$Variant = 'cu128'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  throw 'Project .venv is missing. Run install.ps1 or create .venv before running this script.'
}
$indexUrls = @{
  cu128 = 'https://download.pytorch.org/whl/cu128'
  cpu = 'https://download.pytorch.org/whl/cpu'
}
$torchVersions = @{
  cu128 = '2.9.1'
  cpu = '2.14.1'
}
$index = $indexUrls[$Variant]
$version = $torchVersions[$Variant]
if ($Variant -eq 'cu128') { $expected = '+cu128' } else { $expected = '+cpu' }
$check = @'
import importlib.metadata
try:
    print(importlib.metadata.version('torch'))
except importlib.metadata.PackageNotFoundError:
    print('missing')
'@
$current = & $python -c $check
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the project environment.' }
if ($LASTEXITCODE -eq 0 -and $current -eq "$version$expected") {
  Write-Host "Matching torch $current is already installed; skipping package installation."
} else {
  # Windows locks PyTorch DLLs while a project Python process is running.
  $venv = (Split-Path -Parent (Split-Path -Parent $python)) + '\'
  try {
    $running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop | Where-Object {
      ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($venv, [StringComparison]::OrdinalIgnoreCase)) -or
      ($_.CommandLine -and $_.CommandLine.Replace('/', '\').IndexOf($venv, [StringComparison]::OrdinalIgnoreCase) -ge 0)
    })
  } catch {
    throw 'Could not inspect running project Python processes. Stop the project Python services before changing PyTorch.'
  }
  if ($running.Count) {
    $ids = ($running | ForEach-Object { $_.ProcessId }) -join ', '
    throw "Project Python processes are still running (PID: $ids). Stop the backend and training processes before changing PyTorch; otherwise Windows may lock DLLs and leave the installation incomplete."
  }
  Write-Host "Installing torch==$version$expected from the official $Variant index into .venv only."
  & $python -m pip --isolated install "torch==$version$expected" --index-url $index
  if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed; no environment variables or drivers were changed." }
}

$probe = "import json; from backend.cuda_environment import inspect_environment; print(json.dumps(inspect_environment(full=True), ensure_ascii=False))"
$json = & $python -c $probe
if ($LASTEXITCODE -ne 0) { throw 'Post-install CUDA diagnostics failed.' }
$report = $json | ConvertFrom-Json
Write-Host "PyTorch: $($report.torch.version); CUDA build: $($report.torch.cudaBuild); status: $($report.status)"
if ($Variant -eq 'cu128' -and $report.status -ne 'available') {
  throw "cu128 wheel installed but CUDA is not available ($($report.status)). No driver or environment variable was changed."
}
if (-not $report.torch.installed -or $report.torch.version -ne "$version$expected") {
  throw 'Post-install PyTorch version or import validation failed.'
}
if ($Variant -eq 'cpu' -and $report.torch.cudaBuild) {
  throw 'CPU selection unexpectedly reported a CUDA build.'
}
Write-Host 'Setup completed. Run the backend CUDA tests and an explicit smoke probe before training.'

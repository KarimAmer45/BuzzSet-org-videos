param(
  [string]$Python = "python",
  [string]$OutputPath = "weights\yolovpp_swin_tiny.pth"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -ne "") {
  New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
  Write-Host "Checkpoint already exists: $OutputPath"
  exit 0
}

Write-Host "Installing gdown if needed..."
& $Python -m pip install gdown

$fileId = "1pCIWAK6cy-BHhDVywmPb1LuuQHzNXdT2"
Write-Host "Downloading YOLOV++ SwinTiny checkpoint to $OutputPath ..."
& $Python -m gdown $fileId -O $OutputPath
if ($LASTEXITCODE -ne 0) {
  throw "gdown failed while downloading checkpoint id $fileId."
}

if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
  throw "Checkpoint download failed: $OutputPath was not created."
}

$size = (Get-Item -LiteralPath $OutputPath).Length
if ($size -lt 1000000) {
  throw "Checkpoint download looks too small ($size bytes). Delete $OutputPath and retry."
}

Write-Host "Checkpoint ready: $OutputPath"

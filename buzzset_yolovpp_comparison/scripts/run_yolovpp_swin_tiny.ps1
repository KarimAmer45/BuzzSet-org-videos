param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$CheckpointPath = "weights\yolovpp_swin_tiny.pth",
  [int]$BatchSize = 16,
  [int]$Devices = 1,
  [switch]$NoFp16,
  [switch]$SkipDependencyInstall,
  [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$project = (Resolve-Path $ProjectRoot).Path

Write-Host "Step 1/4: preparing BuzzSet annotations"
& powershell -ExecutionPolicy Bypass -File (Join-Path $project "scripts\prepare_buzzset.ps1") -Python $Python
if ($LASTEXITCODE -ne 0) {
  throw "Annotation preparation failed."
}

if (-not $SkipDependencyInstall) {
  Write-Host "Step 2/4: ensuring training dependencies"
  & $Python -c "import torch" 2>$null
  $torchInstalled = ($LASTEXITCODE -eq 0)
  $depArgs = @(
    "-Python", $Python,
    "-YoloVRoot", $YoloVRoot,
    "-ProjectRoot", $ProjectRoot
  )
  if ($torchInstalled) {
    $depArgs += "-SkipTorch"
  }
  & powershell -ExecutionPolicy Bypass -File (Join-Path $project "scripts\install_training_deps.ps1") @depArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
  }
}
else {
  Write-Host "Step 2/4: dependency install skipped"
}

Write-Host "Step 3/4: ensuring YOLOV++ SwinTiny checkpoint"
& powershell -ExecutionPolicy Bypass -File (Join-Path $project "scripts\download_yolovpp_swin_tiny.ps1") `
  -Python $Python `
  -OutputPath $CheckpointPath
if ($LASTEXITCODE -ne 0) {
  throw "Checkpoint download failed."
}
$CheckpointPath = (Resolve-Path -LiteralPath $CheckpointPath).Path

Write-Host "Step 4/4: starting YOLOV++ SwinTiny training"
$trainArgs = @(
  "-Python", $Python,
  "-YoloVRoot", $YoloVRoot,
  "-ProjectRoot", $ProjectRoot,
  "-Ckpt", $CheckpointPath,
  "-BatchSize", "$BatchSize",
  "-Devices", "$Devices"
)

if (-not $NoFp16) {
  $trainArgs += "-Fp16"
}

if ($ExtraArgs.Count -gt 0) {
  $trainArgs += "-ExtraArgs"
  $trainArgs += $ExtraArgs
}

& powershell -ExecutionPolicy Bypass -File (Join-Path $project "scripts\train_yolovpp_swin_tiny.ps1") @trainArgs

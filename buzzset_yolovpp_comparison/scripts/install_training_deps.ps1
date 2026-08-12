param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
  [switch]$SkipTorch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path ".").Path
$yolov = (Resolve-Path $YoloVRoot).Path
$project = (Resolve-Path $ProjectRoot).Path
$requirements = Join-Path $project "requirements-training.txt"

if (-not $SkipTorch) {
  & $Python -m pip install torch torchvision --index-url $TorchIndexUrl
}

& $Python -m pip install -r $requirements

Push-Location $yolov
try {
  & $Python -m pip install -e . --no-deps
}
finally {
  Pop-Location
  Set-Location $workspace
}


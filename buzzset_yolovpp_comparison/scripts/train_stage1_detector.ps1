param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$Exp = "",
  [string]$Ckpt = "weights\yolovpp_swin_tiny.pth",
  [int]$BatchSize = 16,
  [int]$Devices = 1,
  [int]$MaxEpoch = 0,
  [switch]$Fp16,
  [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path ".").Path
$yolov = (Resolve-Path $YoloVRoot).Path
$project = (Resolve-Path $ProjectRoot).Path
if ($Exp -ne "") {
  $exp = (Resolve-Path -LiteralPath $Exp).Path
}
else {
  $exp = Join-Path $project "exps\buzzset_stage1_swin_tiny_det.py"
}

$env:YOLOV_ROOT = $yolov
$env:PYTHONPATH = "$project;$yolov;$env:PYTHONPATH"

if ($Ckpt -eq "") {
  Write-Warning "No checkpoint given. The Swin backbone will be randomly initialized, which is not a useful detector. Pass -Ckpt with the pretrained YOLOV++ SwinTiny weights."
}
elseif (-not (Test-Path -LiteralPath $Ckpt -PathType Leaf)) {
  throw "Checkpoint file not found: $Ckpt. Download it first (scripts\download_yolovpp_swin_tiny.ps1) or pass a real .pth path."
}
else {
  $Ckpt = (Resolve-Path -LiteralPath $Ckpt).Path
}

$cmd = @("tools\train.py", "-f", $exp, "-d", "$Devices", "-b", "$BatchSize")
if ($Fp16) {
  $cmd += "--fp16"
}
if ($Ckpt -ne "") {
  $cmd += @("-c", $Ckpt)
}
if ($MaxEpoch -gt 0) {
  $cmd += @("max_epoch", "$MaxEpoch")
}
$cmd += $ExtraArgs

Push-Location $yolov
try {
  & $Python @cmd
}
finally {
  Pop-Location
  Set-Location $workspace
}

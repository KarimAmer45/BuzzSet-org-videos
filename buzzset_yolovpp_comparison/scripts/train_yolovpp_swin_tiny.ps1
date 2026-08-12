param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$Ckpt = "",
  [string]$RunName = "",
  [int]$MaxEpoch = 0,
  [int]$BatchSize = 16,
  [int]$Devices = 1,
  [switch]$Fp16,
  [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path ".").Path
$yolov = (Resolve-Path $YoloVRoot).Path
$project = (Resolve-Path $ProjectRoot).Path
$exp = Join-Path $project "exps\buzzset_yolovpp_swin_tiny.py"

$env:YOLOV_ROOT = $yolov
$env:PYTHONPATH = "$project;$yolov;$env:PYTHONPATH"

if ($Ckpt -eq "") {
  Write-Warning "No checkpoint was provided. YOLOV++ fine-tuning is normally meaningful only from a pretrained/paper checkpoint."
}
elseif (-not (Test-Path -LiteralPath $Ckpt -PathType Leaf)) {
  throw "Checkpoint file not found: $Ckpt. Replace -Ckpt with a real .pth path, or omit -Ckpt for a wiring-only smoke test."
}
else {
  $Ckpt = (Resolve-Path -LiteralPath $Ckpt).Path
}

$cmd = @("tools\vid_train.py", "-f", $exp, "-d", "$Devices", "-b", "$BatchSize")
if ($Fp16) {
  $cmd += "--fp16"
}
if ($Ckpt -ne "") {
  $cmd += @("-c", $Ckpt)
}
if ($RunName -ne "") {
  $cmd += @("exp_name", $RunName)
}
if ($MaxEpoch -gt 0) {
  $cmd += @("max_epoch", "$MaxEpoch")
}
$cmd += $ExtraArgs

$pythonExitCode = 1
Push-Location $yolov
try {
  & $Python @cmd
  $pythonExitCode = $LASTEXITCODE
}
finally {
  Pop-Location
  Set-Location $workspace
}

if ($pythonExitCode -ne 0) {
  exit $pythonExitCode
}

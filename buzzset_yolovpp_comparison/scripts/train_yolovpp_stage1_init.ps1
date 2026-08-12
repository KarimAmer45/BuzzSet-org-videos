param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$Stage1Ckpt = "",
  [string]$RunName = "buzzset_yolovpp_swin_tiny_stage1_init",
  [int]$MaxEpoch = 20,
  [int]$BatchSize = 16,
  [int]$Devices = 1,
  [switch]$NoFp16,
  [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$project = (Resolve-Path $ProjectRoot).Path
if ($Stage1Ckpt -eq "") {
  $Stage1Ckpt = Join-Path $project "runs\buzzset_stage1_swin_tiny_det\best_ckpt.pth"
}

if (-not (Test-Path -LiteralPath $Stage1Ckpt -PathType Leaf)) {
  throw "Stage-1 detector checkpoint not found: $Stage1Ckpt. Run train_stage1_detector.ps1 first, or pass -Stage1Ckpt."
}
$Stage1Ckpt = (Resolve-Path -LiteralPath $Stage1Ckpt).Path

$trainScript = Join-Path $project "scripts\train_yolovpp_swin_tiny.ps1"
$stage2Args = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $trainScript,
  "-Python", $Python,
  "-YoloVRoot", $YoloVRoot,
  "-ProjectRoot", $ProjectRoot,
  "-Ckpt", $Stage1Ckpt,
  "-RunName", $RunName,
  "-MaxEpoch", "$MaxEpoch",
  "-BatchSize", "$BatchSize",
  "-Devices", "$Devices"
)
if (-not $NoFp16) { $stage2Args += "-Fp16" }
if ($ExtraArgs.Count -gt 0) { $stage2Args += @("-ExtraArgs") + $ExtraArgs }

Write-Host "Starting YOLOV++ improvement run from stage-1 detector:"
Write-Host "  checkpoint: $Stage1Ckpt"
Write-Host "  run name:   $RunName"
Write-Host "  max epoch:  $MaxEpoch"
Write-Host "  batch size: $BatchSize"

& powershell @stage2Args
if ($LASTEXITCODE -ne 0) {
  throw "YOLOV++ stage-1-initialized training failed."
}

Write-Host "Improvement run complete."
Write-Host "Output folder: $(Join-Path $project "runs\$RunName")"

param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [string]$Stage1Ckpt = "weights\yolovpp_swin_tiny.pth",
  [int]$BatchSize = 16,
  [int]$Devices = 1,
  [switch]$NoFp16,
  [switch]$SkipStage1,
  [string[]]$Stage1ExtraArgs = @(),
  [string[]]$Stage2ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$project = (Resolve-Path $ProjectRoot).Path
$stage1Best = Join-Path $project "runs\buzzset_stage1_swin_tiny_det\best_ckpt.pth"

if (-not $SkipStage1) {
  Write-Host "Stage 1/2: fine-tuning the single-frame YOLOX Swin-Tiny detector on BuzzSet frames"
  $stage1Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $project "scripts\train_stage1_detector.ps1"),
    "-Python", $Python,
    "-YoloVRoot", $YoloVRoot,
    "-ProjectRoot", $ProjectRoot,
    "-Ckpt", $Stage1Ckpt,
    "-BatchSize", "$BatchSize",
    "-Devices", "$Devices"
  )
  if (-not $NoFp16) { $stage1Args += "-Fp16" }
  if ($Stage1ExtraArgs.Count -gt 0) { $stage1Args += @("-ExtraArgs") + $Stage1ExtraArgs }
  & powershell @stage1Args
  if ($LASTEXITCODE -ne 0) { throw "Stage 1 detector training failed." }
}
else {
  Write-Host "Stage 1/2: skipped (using existing stage-1 checkpoint)"
}

if (-not (Test-Path -LiteralPath $stage1Best -PathType Leaf)) {
  throw "Stage-1 checkpoint not found: $stage1Best. Run stage 1 first (omit -SkipStage1)."
}
$stage1Best = (Resolve-Path -LiteralPath $stage1Best).Path

Write-Host "Stage 2/2: training YOLOV++ temporal aggregation on top of the BuzzSet detector"
$stage2Args = @(
  "-ExecutionPolicy", "Bypass",
  "-File", (Join-Path $project "scripts\train_yolovpp_swin_tiny.ps1"),
  "-Python", $Python,
  "-YoloVRoot", $YoloVRoot,
  "-ProjectRoot", $ProjectRoot,
  "-Ckpt", $stage1Best,
  "-BatchSize", "$BatchSize",
  "-Devices", "$Devices"
)
if (-not $NoFp16) { $stage2Args += "-Fp16" }
if ($Stage2ExtraArgs.Count -gt 0) { $stage2Args += @("-ExtraArgs") + $Stage2ExtraArgs }
& powershell @stage2Args
if ($LASTEXITCODE -ne 0) { throw "Stage 2 aggregation training failed." }

Write-Host "Two-stage training complete."
Write-Host "Stage-1 detector: $stage1Best"
Write-Host "Stage-2 YOLOV++:  $(Join-Path $project 'runs\buzzset_yolovpp_swin_tiny\best_ckpt.pth')"

param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$ProjectRoot = "buzzset_yolovpp_comparison",
  [Parameter(Mandatory = $true)][string]$Ckpt,
  [ValidateSet("valid", "test")][string]$EvalSplit = "test",
  [int]$BatchSize = 16,
  [int]$Workers = 4,
  [switch]$Fp16,
  [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path ".").Path
$yolov = (Resolve-Path $YoloVRoot).Path
$project = (Resolve-Path $ProjectRoot).Path
$exp = Join-Path $project "exps\buzzset_yolovpp_swin_tiny.py"
$evalScript = Join-Path $project "scripts\eval_yolov_exp.py"

$env:YOLOV_ROOT = $yolov
$env:PYTHONPATH = "$project;$yolov;$env:PYTHONPATH"

if (-not (Test-Path -LiteralPath $Ckpt -PathType Leaf)) {
  throw "Checkpoint file not found: $Ckpt"
}
$Ckpt = (Resolve-Path -LiteralPath $Ckpt).Path

$cmd = @(
  $evalScript,
  "--yolov-root", $yolov,
  "-f", $exp,
  "-c", $Ckpt,
  "--eval-split", $EvalSplit,
  "--batch-size", "$BatchSize",
  "--workers", "$Workers"
)
if ($Fp16) {
  $cmd += "--fp16"
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

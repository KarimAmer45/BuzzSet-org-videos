# Stage-2 context-frame ablation (TransVOD Fig. b style).
# Trains the YOLOV++ temporal head (Stage-1 frozen) for several context-frame
# counts and leaves one run per setting. Evaluate each with tools/eval_stratified.py.
#
# The number of context frames = global reference frames (gframe); it is passed
# through the CONTEXT_FRAMES env var, and the batch size must equal it.
param(
  [string]$Python = "python",
  [string]$YoloVRoot = "YOLOV-master",
  [string]$Exp = "buzzset_yolovpp_comparison\exps\buzzset_v2_yolovpp_swin_tiny.py",
  [string]$Stage1Ckpt = "buzzset_yolovpp_comparison\runs\buzzset_v2_det_swin_tiny\best_ckpt.pth",
  [int[]]$ContextFrames = @(4, 8, 16, 32),
  [switch]$NoFp16
)
$ErrorActionPreference = "Stop"
$fp16 = if ($NoFp16) { @() } else { @("--fp16") }

foreach ($N in $ContextFrames) {
  Write-Host "==================  CONTEXT_FRAMES = $N  (batch = gframe = $N)  =================="
  $env:CONTEXT_FRAMES = "$N"
  & $Python (Join-Path $YoloVRoot "tools\vid_train.py") -f $Exp -c $Stage1Ckpt -b $N @fp16
  if ($LASTEXITCODE -ne 0) { throw "training failed for CONTEXT_FRAMES=$N" }
}
Write-Host "Done. For each runs\buzzset_v2_yolovpp_swin_tiny_ctx<N>\best_ckpt.pth:"
Write-Host "  1) dump predictions on valid/test (YOLOV tools\vid_eval.py --save_result)"
Write-Host "  2) python buzzset_yolovpp_comparison\tools\eval_stratified.py --gt <ann> --dt <pred> --conf 0.30"

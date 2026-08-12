# Download the Swin-Base weights for the backbone experiment, from the YOLOV
# model zoo (same source as the Swin-Tiny checkpoint). Mirrors download_yolovpp_swin_tiny.ps1.
#
#   Stage-1 detector init : YOLOX-SwinBase  (Drive folder) -> weights\yolox_swin_base\
#   Stage-2 reference/init: YOLOV++ SwinBase (Drive file)  -> weights\yolovpp_swin_base.pth
param(
  [string]$Python = "python",
  [string]$WeightsDir = "weights"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null
& $Python -m pip install gdown | Out-Null

# --- YOLOV++ SwinBase (Stage-2 VID checkpoint) ---
$vidOut = Join-Path $WeightsDir "yolovpp_swin_base.pth"
if (-not (Test-Path -LiteralPath $vidOut)) {
  Write-Host "Downloading YOLOV++ SwinBase -> $vidOut"
  & $Python -m gdown "1RGb499EBcSQjWDxu6KkvN4Tr1wSc6SHb" -O $vidOut
} else { Write-Host "exists: $vidOut" }

# --- YOLOX-SwinBase (Stage-1 detector weights, published as a Drive folder) ---
$detDir = Join-Path $WeightsDir "yolox_swin_base"
if (-not (Test-Path -LiteralPath $detDir)) {
  Write-Host "Downloading YOLOX-SwinBase detector folder -> $detDir"
  & $Python -m gdown --folder "https://drive.google.com/drive/folders/1K5897iM2zzN4kcj8qdK3z_FtvW9f3kHN" -O $detDir
} else { Write-Host "exists: $detDir" }

Write-Host ""
Write-Host "Done. Use as the -c checkpoint:"
Write-Host "  Stage-1 Swin-Base : -Ckpt weights\yolox_swin_base\<file>.pth  (exps\buzzset_v2_det_swin_base.py)"
Write-Host "  Stage-2 Swin-Base : -c    <your trained Stage-1 best_ckpt.pth> (exps\buzzset_v2_yolovpp_swin_base.py)"
Write-Host "If a raw Swin backbone needs adapting to the YOLOV layout, run:"
Write-Host "  python ..\YOLOV-master\tools\convert_swin_weights.py -c <in.pth> -oc <out.pth>"

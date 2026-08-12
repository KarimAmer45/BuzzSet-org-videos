param(
  [string]$Python = "python",
  [string]$SourceRoot = "BuzzSet-org-videos\BuzzSetV2_split",
  [string]$OutputDir = "buzzset_yolovpp_comparison\generated",
  [int]$GFrame = 16
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

& $Python "buzzset_yolovpp_comparison\scripts\convert_buzzset_to_yolov.py" `
  --source-root $SourceRoot `
  --output-dir $OutputDir

& $Python "buzzset_yolovpp_comparison\scripts\audit_buzzset_yolov.py" `
  --data-root $SourceRoot `
  --ann-dir (Join-Path $OutputDir "annotations") `
  --gframe $GFrame


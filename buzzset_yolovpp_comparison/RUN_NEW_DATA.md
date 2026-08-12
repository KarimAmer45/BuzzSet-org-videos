# BuzzSet Challenge (video) — single-frame milestone runs

The new **BuzzSet Challenge** video dataset is the correct data: per-video folders
of consecutive frames (an annotated keyframe + 5 preceding frames), **4 classes**
(`bee, bumblebee, hoverfly, moth`). For the milestone we train two single-frame
reference detectors on the **keyframes**. The temporal YOLOV++ run is deferred
(per Lennart) but the data now supports it.

Dataset on disk: `data\` (at the project root) with
`train/` (77 videos, 5,275 keyframes / 10,613 boxes) and `valid/` (21 videos,
932 keyframes / 1,086 boxes). `test_devphase/` has **no annotations yet** — drop
in Lennart's `test.json` and rename `test_devphase/` → `test/` before any test eval.

## Already prepared (no GPU needed)

- Keyframe-only 4-class COCO: `generated_v2/annotations/{train,valid}_keyframes.json`
- RF-DETR-format annotations placed in the dataset: `data\{train,valid}\_annotations.coco.json`
- YOLOX exp: `exps/buzzset_v2_det_swin_tiny.py` (4-class, 576 px, rebalanced alpha=0.3)
- RF-DETR script: `rfdetr/train_rfdetr.py`

---

## Track 1 — YOLOX-Swin detector (reuses the working pipeline)

This is the fastest path; it reuses the same scripts/dataset as before. From
`F:\BuzzSet-org-videos`:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_stage1_detector.ps1 `
  -Exp ".\buzzset_yolovpp_comparison\exps\buzzset_v2_det_swin_tiny.py" `
  -Ckpt "F:\BuzzSet-org-videos\weights\yolovpp_swin_tiny.pth" `
  -BatchSize 16 -Fp16 -MaxEpoch 20
```

- 576 px, mosaic on, class-balanced (alpha=0.3), starts from the pretrained Swin backbone.
- Writes to `runs\buzzset_v2_det_swin_tiny\`; per-epoch validation AP (mAP/AP50/AP75 + per-class) prints in the log.
- The keyframes are essentially the old single-frame set minus `other_insect`, so expect a number in the mid-20s–30 mAP range. Raise `input_size`/`test_size` to 768 in the exp once a stable GPU is available.

Evaluate the best checkpoint (validation):

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\eval_yolovpp_swin_tiny.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\buzzset_yolovpp_comparison\runs\buzzset_v2_det_swin_tiny\best_ckpt.pth" `
  -EvalSplit valid -BatchSize 16 -Fp16
```

---

## Track 2 — RF-DETR (Lennart's named reference)

Install once (CUDA GPU with ≥8 GB VRAM for `base`; use `--model small` or `nano` for ~6 GB):

```powershell
pip install rfdetr
```

Train (from `F:\BuzzSet-org-videos`):

```powershell
C:\Users\karim\AppData\Local\Programs\Python\Python312\python.exe `
  .\buzzset_yolovpp_comparison\rfdetr\train_rfdetr.py `
  --epochs 30 --batch-size 4 --model base
```

The dataset is already in RF-DETR's COCO layout (`train/` + `valid/` each with
`_annotations.coco.json`). Output goes to `runs\rfdetr_v2\`.

**Two things to check on the first run** (RF-DETR is newer than the rest of the stack):
- *Image paths.* Our `file_name`s include the per-video subfolder (e.g.
  `train_video_1/train_video_1_015330.jpg`). RF-DETR's COCO loader should join
  these under `train/`; if it reports missing images, the subfolders are the
  cause and we'll flatten them.
- *Test split.* RF-DETR may expect a `test/` subdir. There's no annotated test
  set yet, so either wait for Lennart's `test.json`, or copy
  `valid\_annotations.coco.json` into a `test/` folder as a temporary placeholder.

---

## Notes

- **Test eval is pending** Lennart's `test.json`. Until then, validation AP is the milestone number (report it as validation, not test).
- **Temporal YOLOV++** is deferred per Lennart (the ImageNet-VID reference data is ~100 GB, too big for Bender right now) — but this dataset finally has the consecutive frames for it, so it's a real follow-up.
- **GPU stability:** keep long runs off a shared/gaming GPU; 576 px is the stable setting locally; both pipelines checkpoint each epoch, so a crash costs at most one epoch.

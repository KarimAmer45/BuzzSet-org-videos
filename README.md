# BuzzSet YOLOV++ — video pollinator detection

Working tree for the **YOLOV++** track of the BuzzSet project: a two-stage
temporal detector (single-frame **YOLOX Swin-Tiny** → **YOLOV++** temporal
aggregation) on the 4-class BuzzSet Challenge video data
(`bee, bumblebee, hoverfly, moth`).

## Layout
```
.
├── data/                        # BuzzSet Challenge (4-class video): train/ valid/ test_devphase/ annotations/
├── buzzset_yolovpp_comparison/  # our code — exps, package, scripts, rfdetr, generated_v2 annotations
│   ├── exps/                    #   training configs (buzzset_v2_det_swin_tiny*.py)
│   ├── buzzset_yolov/           #   dataset wrapper, class-balanced sampling, torch shim
│   ├── scripts/                 #   train / eval runners
│   ├── rfdetr/                  #   RF-DETR track (teammate's model)
│   ├── generated_v2/annotations #   4-class keyframe COCO (train/valid)
│   ├── runs/                    #   training outputs / checkpoints
│   ├── README.md                #   code-level docs
│   └── RUN_NEW_DATA.md          #   full train/eval commands
├── YOLOV-master/                # upstream YOLOV / YOLOX (dependency, provides `yolox`)
├── weights/                     # yolovpp_swin_tiny.pth (pretrained backbone)
└── _archive/                    # non-code: reports, slides, figures, spreadsheets (not needed to train)
```

## Data
`data/` is the **new 4-class** BuzzSet Challenge video set — this is the only data
we train on (the old 5-class set has been removed). Keyframe COCO labels live in
`buzzset_yolovpp_comparison/generated_v2/annotations/{train,valid}_keyframes.json`.
`data/test_devphase/` holds the test images (labels pending from the supervisor).

## Train (quickstart)
From this folder (Windows / PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_stage1_detector.ps1 `
  -Exp ".\buzzset_yolovpp_comparison\exps\buzzset_v2_det_swin_tiny.py" `
  -Ckpt ".\weights\yolovpp_swin_tiny.pth" -BatchSize 16 -Fp16 -MaxEpoch 12
```
The exp defaults to `./data`; set the `BUZZSET_V2_ROOT` env var to override.
Full commands (fast/full configs, evaluation, RF-DETR): see
`buzzset_yolovpp_comparison/RUN_NEW_DATA.md` and `.../README.md`.

## Notes
- **`_archive/`** holds everything non-code (reports, slide decks, spreadsheets,
  figures, talking points, old scratch). Nothing there is needed to train —
  delete the folder if you don't want it.
- Training runs and checkpoints are under `buzzset_yolovpp_comparison/runs/`.

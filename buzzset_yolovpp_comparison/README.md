# BuzzSet YOLOV++ Comparison Project

This folder turns the provided BuzzSet frame annotations into a YOLOV++-ready
video-detection project, while leaving the original data and upstream YOLOV code
unchanged.

It follows the YOLOV++ paper, "Practical Video Object Detection via Feature
Selection and Aggregation" (`arXiv:2407.19650`), and the local `YOLOV-master`
code. See `PAPER_NOTES.md` for the paper details that drove the config choices.

## What Is Included

- `scripts/convert_buzzset_to_yolov.py`: converts each BuzzSet COCO split into
  video-grouped COCO JSON with `videos`, `sid`, `fid`, and `name` fields.
- `scripts/audit_buzzset_yolov.py`: checks generated annotations, missing
  images, duplicate names, sequence lengths, class counts, and frame retention
  for a chosen `gframe`.
- `buzzset_yolov/`: the YOLOV dataset/evaluator adapter used by the experiment.
- `exps/buzzset_yolovpp_swin_tiny.py`: a YOLOV++ Swin-Tiny fine-tuning config
  for the five BuzzSet classes.
- `exps/buzzset_stage1_swin_tiny_det.py`: the stage-1 single-frame YOLOX
  Swin-Tiny detector config (see Two-Stage Training).
- `buzzset_yolov/coco_dataset.py`: single-frame BuzzSet COCO dataset for
  stage 1; `scripts/train_stage1_detector.ps1` and `scripts/run_two_stage.ps1`
  run the two stages.
- `scripts/eval_yolov_exp.py`: an eval entrypoint that respects the experiment
  dataset loader, unlike this repo's stock `tools/vid_eval.py`.
- `results/current_baseline_template.csv`: a place to record your current
  baseline metrics so the comparison script can compute deltas.
- `PAPER_NOTES.md`: short paper-to-project mapping for training/evaluation
  choices.

## Dataset Layout Expected

From the workspace root:

```text
F:\BuzzSet-org-videos
  BuzzSet-org-videos\BuzzSetV2_split
    train\_annotations.coco.json
    valid\_annotations.coco.json
    test\_annotations.coco.json
  YOLOV-master
  buzzset_yolovpp_comparison
```

BuzzSet filenames encode sequence identity:

```text
20230708_plot3_13.12_30674.jpg
^^^^^^^^ ^^^^^ ^^^^^  ^^^^^
date     plot  time   frame index
```

The converter groups frames by `date_plot_time` and sorts them by frame index.

## Prepare The YOLOV++ Annotations

From `F:\BuzzSet-org-videos`:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\prepare_buzzset.ps1
```

Equivalent Python commands:

```powershell
python .\buzzset_yolovpp_comparison\scripts\convert_buzzset_to_yolov.py
python .\buzzset_yolovpp_comparison\scripts\audit_buzzset_yolov.py --gframe 16
```

Generated files are written to:

```text
buzzset_yolovpp_comparison\generated\annotations
```

The local split summary with filename-derived videos is:

```text
split   images  anns   videos  >=16-frame groups  >=16-frame images
train   5401    10930  82      45                 5237
valid   936     1095   21      11                 871
test    2155    2936   29      17                 2080
```

`gframe=16` is the default here because it keeps most frames. To mimic the
paper-style validation setting more closely, try `gframe_val 32`, but note that
short clips are skipped by the YOLOV sampler.

## Two-Stage Training (recommended)

YOLOV++ only fine-tunes the temporal aggregation on top of a single-frame
detector it assumes is already good. The bundled YOLOV++ SwinTiny checkpoint was
trained on ImageNet VID, so its detector has never seen insects; fine-tuning
only the aggregation on top of it leaves the per-frame detector as the
bottleneck. The two-stage flow fixes that:

1. Stage 1: fine-tune the single-frame YOLOX SwinTiny detector on the BuzzSet
   frames, starting from the pretrained checkpoint.
2. Stage 2: run the YOLOV++ aggregation training (the section below) starting
   from the stage-1 detector instead of the ImageNet VID checkpoint.

Stage 1 also produces the single-frame baseline AP to compare the YOLOV++ run
against.

Run both stages from `F:\BuzzSet-org-videos`:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\run_two_stage.ps1 `
  -Stage1Ckpt "F:\BuzzSet-org-videos\weights\yolovpp_swin_tiny.pth" `
  -BatchSize 16
```

Or run the stages separately:

```powershell
# Stage 1: single-frame detector
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_stage1_detector.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\weights\yolovpp_swin_tiny.pth" -BatchSize 16 -Fp16

# Stage 2: YOLOV++ aggregation on top of the stage-1 detector
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_yolovpp_swin_tiny.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\buzzset_yolovpp_comparison\runs\buzzset_stage1_swin_tiny_det\best_ckpt.pth" `
  -BatchSize 16 -Fp16
```

Both stages are now set to the full-resolution push: `768 x 768`, mosaic on with
a 0.5 scale floor (mixup off), stage-1 `max_epoch=25`, stage-2 `max_epoch=30`.
Resolution is the dominant small-object lever here: at 576 about a quarter of the
insect boxes fall under ~12 px after downscaling, and 768 cuts that to ~11%.

Training-time warning: on this GPU the heavier setting was measured at only ~80
of 338 iterations in about two hours (multi-day ETA), so running the fast
baseline first is reasonable. To put stage 1 back on the bounded Windows-friendly
config, set in `exps\buzzset_stage1_swin_tiny_det.py`:

```python
self.input_size = (576, 576)
self.test_size = (576, 576)
self.multiscale_range = 0
self.mosaic_prob = 0.0
self.enable_mixup = False
self.max_epoch = 8
```

and set `self.input_size` / `self.test_size` back to `(576, 576)` in
`exps\buzzset_yolovpp_swin_tiny.py` for the stage-2 run.

## Improvement Run For The 7.77% AP Baseline

The original `buzzset_yolovpp_swin_tiny` run reached `7.77%` validation AP while
starting from the paper checkpoint trained on ImageNet VID. The better next run
starts YOLOV++ from the BuzzSet stage-1 detector checkpoint instead:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_yolovpp_stage1_init.ps1 `
  -BatchSize 16
```

This writes to a separate run folder:

```text
buzzset_yolovpp_comparison\runs\buzzset_yolovpp_swin_tiny_stage1_init
```

Use `-MaxEpoch 30` for a longer attempt after the 20-epoch run finishes.

## Train YOLOV++ Swin-Tiny

Direct run from `F:\BuzzSet-org-videos`:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\run_yolovpp_swin_tiny.ps1
```

That command prepares annotations, ensures training dependencies, downloads the
YOLOV++ SwinTiny checkpoint into `weights\yolovpp_swin_tiny.pth`, and starts
training.

If dependencies are already installed and you only want to run:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\run_yolovpp_swin_tiny.ps1 -SkipDependencyInstall
```

Install YOLOV dependencies in your training environment first.

Do not use YOLOV's full `requirements.txt` on Python 3.12: it pins old ONNX
export packages such as `onnxruntime==1.8.0`, which do not have Python 3.12
wheels and are not needed for training/evaluation here.

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\install_training_deps.ps1
```

The install script installs PyTorch with CUDA 12.8 wheels by default, then
installs the training dependencies from
`buzzset_yolovpp_comparison\requirements-training.txt`, and finally installs
YOLOV editable with `--no-deps` so the old ONNX pins are skipped.
The training dependency list includes `timm`, which YOLOV's Swin backbone needs.

If PyTorch is already installed in your environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\install_training_deps.ps1 -SkipTorch
```

For CPU-only PyTorch, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\install_training_deps.ps1 -TorchIndexUrl "https://download.pytorch.org/whl/cpu"
```

Then run from the workspace root:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_yolovpp_swin_tiny.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\weights\yolovpp_swin_tiny.pth" `
  -BatchSize 16 `
  -Devices 1 `
  -Fp16
```

The checkpoint is strongly recommended. The YOLOV++ config freezes most of the
backbone/stem the same way the upstream experiment does, so training from random
weights is not a meaningful comparison.

Useful overrides can be appended after `-ExtraArgs`, for example:

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\train_yolovpp_swin_tiny.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\weights\yolovpp_swin_tiny.pth" `
  -ExtraArgs max_epoch 40 basic_lr_per_img 0.000015625 eval_name test
```

## Evaluate

```powershell
powershell -ExecutionPolicy Bypass -File .\buzzset_yolovpp_comparison\scripts\eval_yolovpp_swin_tiny.ps1 `
  -Ckpt "F:\BuzzSet-org-videos\buzzset_yolovpp_comparison\runs\buzzset_yolovpp_swin_tiny\best_ckpt.pth" `
  -EvalSplit test `
  -BatchSize 16 `
  -Fp16
```

The evaluator writes YOLOV's usual `refined_pred.json`, `gt_refined.json`, and
log output in the current YOLOV working directory. The AP table uses BuzzSet's
five classes: `bee`, `bumblebee`, `hoverfly`, `other_insect`, and `moth`.

By default the evaluator now scores the full split, including clips shorter than
`gframe`; pass `--no-formal` through `-ExtraArgs` to skip them (matching the
faster per-epoch eval used during training). To read the per-class AP next to how
many instances back each class, run:

```powershell
python .\buzzset_yolovpp_comparison\scripts\summarize_class_support.py
```

Classes with very little validation support (here `bumblebee` and
`other_insect`) give noisy AP that drags the unweighted mean down.

## Compare Against Your Current Result

Fill in `results/current_baseline_template.csv` with the metrics you have
already reached, then create a similar CSV for the YOLOV++ run and compare:

```powershell
python .\buzzset_yolovpp_comparison\scripts\compare_metrics.py `
  --baseline .\buzzset_yolovpp_comparison\results\current_baseline_template.csv `
  --candidate .\buzzset_yolovpp_comparison\results\yolovpp_run.csv
```

The comparison script matches rows by `split` and `metric`, then reports
candidate minus baseline deltas.

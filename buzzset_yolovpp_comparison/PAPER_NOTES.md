# Paper Notes For This Project

Reference: `arXiv:2407.19650` (also published in IJCV, 2026).

Paper identity:

- Title: Practical Video Object Detection via Feature Selection and Aggregation.
- Venue/version: International Journal of Computer Vision, 2026, 134:95.
- DOI: `10.1007/s11263-025-02700-3`.
- Authors: Yuheng Shi, Tong Zhang, Xiaojie Guo.

## Method Mapping

The paper's YOLOV++ pipeline first condenses candidate foreground features from
dense one-stage detector predictions, then aggregates selected reference-frame
features. The upstream `YOLOV-master/exps/yolov++/v++_SwinTiny_decoupleReg.py`
already implements that comparison target with:

- separate feature aggregation paths for classification and regression;
- multi-head attention over selected candidate/reference features;
- proposal filtering with top-K/NMS/confidence style feature selection;
- a decoupled regression branch.

The BuzzSet experiment inherits that upstream config and changes only the
dataset, class count, output paths, and practical schedule defaults.

## Paper Settings Reflected Here

- Input/test resolution is kept at `576 x 576`, matching the YOLOV++ experiments.
- `gframe=16` is the default training/evaluation frame group size. The paper
  states that the feature aggregation module uses 16 frames.
- Test confidence remains `0.001`.
- Final detection NMS remains `0.5`.
- The base learning-rate scale inherited from YOLOV is
  `0.002 / 64` per image, so `BatchSize 16` gives `5e-4`, matching the paper's
  FAM fine-tuning learning-rate scale for most models.
- A pretrained YOLOV++ or compatible detector checkpoint is important. The paper
  fine-tunes aggregation on top of initialized detector weights; random
  initialization is not a fair comparison.

## Project-Specific Deviations

BuzzSet is much smaller and uses filename-derived video groups rather than
ImageNet VID sequences. The default `max_epoch=20` is a practical starter
setting, not the paper's full 150K-iteration ImageNet VID schedule. For a closer
paper-style schedule, estimate the iterations per epoch after conversion and set
`max_epoch` accordingly through `-ExtraArgs`.

The provided BuzzSet split has short clips. With `gframe=16`, retained frames
are:

```text
train: 5237 / 5401
valid: 871 / 936
test:  2080 / 2155
```

Using `gframe_val=32` keeps fewer short sequences, so it is available as an
override rather than the default.


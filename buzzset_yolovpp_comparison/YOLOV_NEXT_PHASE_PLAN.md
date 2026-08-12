# YOLOV++ — next-phase plan & ML-correctness check
Response to Lennart's email, filtered to the **YOLOV** track. Files referenced
below now exist in this repo.

## 1. ML-correctness of current results (his four questions)

**Hyperparameters — Stage-1 detector** (`exps/buzzset_v2_det_swin_tiny.py`, base `swin_tiny_base.py`):
- backbone **Swin-Tiny**; optimizer **AdamW**, `weight_decay 0.05`, `basic_lr_per_img = 0.0001/18 × batch`, scheduler `yoloxwarmcos`, warmup 1 epoch
- input **576** (fast 512), batch **16**, fp16; mosaic p=1.0 + multiscale (fast: off), mixup off
- **class-balanced oversampling**, inverse-frequency, `alpha=0.3`
- eval: `test_conf=0.001`, `nmsthre=0.65`, `eval_interval=1`; `max_epoch` 12–20 (fast 8)

**Hyperparameters — Stage-2 YOLOV++** (`exps/buzzset_v2_yolovpp_swin_tiny.py`, base `v++_SwinTiny_decoupleReg.py`):
- **Backbone + Stage-1 detection head are FROZEN by the base** — only the temporal
  aggregation head trains. (This *is* Lennart's "frozen Stage-1 → train only the
  aggregation part"; it's built into `get_model()`, verified.)
- optimizer **SGD** (momentum 0.9, nesterov), `weight_decay 5e-4`, `basic_lr_per_img = 0.002/64 × batch`, `stem_lr_ratio 0.1`, scheduler `yoloxwarmcos`, **EMA on**
- context frames = **global reference frames** `gframe` (default 16), `gmode=True`, `lframe=0`; **batch size must equal `lframe+gframe`**
- input 576, warmup 1, no_aug 2

**Best checkpoint:** validation is run every epoch (`eval_interval=1`); we keep
`best_ckpt.pth` = highest **validation** mAP(.50:.95). Correct.

**Validation set:** yes — used for per-epoch eval and checkpoint selection.

**Which set we report:** **currently validation.** The 4-class *test* labels are
not populated yet (`data/annotations/test_devphase.json` has images but no
annotations). → **Fix:** keep selection on val, but report **test** metrics once
the test labels land. State the seed; run **multiple seeds** for the headline.

## 2. Main experiment — Single-frame Stage-1 on the **test** set
Evaluate the Stage-1 detector on test and slice it:
- **mAP@50 by** class · object size (COCO small/med/large) · **occlusion** · **blur**
- **FP/FN per class** at a fixed confidence (e.g. 0.3, IoU≥0.5)
- TP/FP/FN **crops** for eyeballing (hoverfly look-alikes, the false-bee positives)

Tool: **`tools/eval_stratified.py`** (tested). Example:
```bash
# dump predictions on the split, then stratify
python ../YOLOV-master/tools/eval.py -f exps/buzzset_v2_det_swin_tiny.py \
  -c runs/buzzset_v2_det_swin_tiny/best_ckpt.pth -b 16 --fp16 --conf 0.001 --save_result
python tools/eval_stratified.py --gt ../data/annotations/valid.json \
  --dt runs/buzzset_v2_det_swin_tiny/predictions.json \
  --img-root ../data/valid --conf 0.30 --dump-crops out/crops
```
(Swap `valid.json` → the test annotations once available.)

## 3. Stage-2 — frozen Stage-1 backbone + context-frame ablation
- Config: **`exps/buzzset_v2_yolovpp_swin_tiny.py`** (new 4-class data, backbone
  frozen by base, `CONTEXT_FRAMES` env → `gframe`).
- Sweep: **`scripts/run_context_frame_ablation.ps1`** (context frames 4/8/16/32;
  batch = gframe). Report mAP@50 by class/size/attribute per setting via the tool above.
- Start each Stage-2 run from the trained Stage-1 detector (`-c .../best_ckpt.pth`).

## 4. Backbone experiment — Swin-Tiny vs Swin-Base
- **Swin-Base Stage-1** ready: **`exps/buzzset_v2_det_swin_base.py`**; **Swin-Base Stage-2** ready: **`exps/buzzset_v2_yolovpp_swin_base.py`** (base
  `v++_SwinBaseX_decoupleReg.py`, `backbone_name='Swin_Base'`).
- Note: our **Stage-1 is already Swin-Tiny**. A Swin-**Base** *Stage-1 detector*
  had no ready base in this checkout, so we built one:
  **`exps/buzzset_v2_det_swin_base.py`** (Swin-Base backbone mirroring
  `v++_SwinBaseX` + the YOLOX detector head). It needs **Swin-Base pretrained
  weights** for `-c` — grab those in the YOLOV meeting.
- Deliverable: the 2×2 (Tiny/Base × Stage-1/Stage-2-best-ctx) and the temporal-gain-
  vs-backbone plot (mirror TransVOD Fig. b).


## Results so far — Stage-1 detector (new 4-class data, validation, best epoch)

| Metric | Swin-Tiny | Swin-Base | Δ |
|---|---|---|---|
| mAP (.50:.95) | 31.6 | **36.0** | **+4.4** |
| AP50 | 54.4 | 56.8 | +2.4 |
| AP75 | 32.5 | 40.7 | +8.2 |
| AR@100 | 40.0 | 49.8 | +9.8 |
| AP (small) | 0.9 | 0.6 | ~0 (both) |

Swin-Base per-class AP: bee 49.8, bumblebee 50.8, moth 37.3, **hoverfly 6.0**.
Chart: `results/stage1_backbone_comparison.png`.

**Findings:** (1) the stronger backbone is the big lever (+4.4 mAP, and larger gains
at AP75 / AR) — matches Lennart's Fig-b thesis. (2) **Small-object AP is ~0 for both
backbones** — capacity doesn't help; this is a resolution problem (576 downscale), the
target for a tiling / native-res experiment. Hoverfly remains the hardest class.

## 5. Logistics
- Propose a meeting **next week** + **Mon 24 Aug** (Lennart on vacation 17–21 Aug).
- Slides review **27 Aug** (one week before the final).

### Open dependencies
1. **Test labels** for `data/annotations/test_devphase.json` (§1, §2 final numbers).
2. **Occlusion/blur attributes — CONFIRMED present** in `data/annotations/*.json`:
   `occluded ∈ {1,2,3}`, `blur ∈ {0,1}`. `tools/eval_stratified.py` slices mAP@50 by
   these out of the box (no change needed).
3. Swin-Base pretrained weights for the backbone contrast.

> The old `exps/buzzset_yolovpp_swin_tiny.py` points at the removed 5-class data
> (`BuzzSetV2_split` / `generated/`) and is superseded by
> `buzzset_v2_yolovpp_swin_tiny.py`.

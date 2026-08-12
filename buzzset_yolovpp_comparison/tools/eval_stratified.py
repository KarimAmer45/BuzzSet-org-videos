#!/usr/bin/env python3
"""Stratified detection analysis for the BuzzSet YOLOV / TransVOD experiments.

Given COCO ground truth and a COCO-format detections file, report **mAP@50**:
  * overall and per class
  * by object size (COCO small / medium / large)
  * by attribute (occlusion, blur) if the annotations carry them
and, at a fixed confidence threshold, the **FP / FN / TP per class** (IoU>=0.5),
with optional TP/FP/FN **crop dumps** for eyeballing.

This directly implements the "Main experiment" analysis in the supervisor's plan:
evaluate mAP@50 w.r.t. class / object size / occlusion / blur, plus FP/FN at a
fixed confidence, and look at concrete object crops.

Usage
-----
    python tools/eval_stratified.py \
        --gt  generated_v2/annotations/test_keyframes.json \
        --dt  runs/<exp>/test_predictions.json \
        --img-root /path/to/BuzzSet_challenge/test \
        --conf 0.30 --iou 0.5 --dump-crops out/crops --max-crops 40

`--dt` is a standard COCO results list: [{"image_id","category_id","bbox":[x,y,w,h],"score"}, ...].
Attribute fields are read from ann["attributes"][key] or ann[key]; a slice is
skipped (with a note) if the field is absent.
"""
import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:  # pragma: no cover
    raise SystemExit("pycocotools is required: pip install pycocotools") from e

ATTR_KEYS = ("occluded", "blur")  # extend if the dataset exposes more


# ----------------------------- helpers -------------------------------------
def get_attr(ann, key):
    a = ann.get("attributes")
    if isinstance(a, dict) and key in a:
        return a[key]
    return ann.get(key)


def coco_from_dict(d):
    c = COCO()
    c.dataset = d
    c.createIndex()
    return c


def ap50_per_class(coco_gt, coco_dt, area="all"):
    """Per-class AP@IoU=0.50 for a given COCO area label."""
    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.iouThrs = np.array([0.50])
    E.params.maxDets = [100]
    E.evaluate()
    E.accumulate()
    prec = E.eval["precision"]  # [T, R, K, A, M]
    a = E.params.areaRngLbl.index(area)
    out = {}
    for ki, cid in enumerate(E.params.catIds):
        p = prec[0, :, ki, a, -1]
        p = p[p > -1]
        out[cid] = float(np.mean(p)) if p.size else float("nan")
    return out


def filter_gt_by_attr(gt_dict, key, value):
    """Copy GT but mark anns whose attribute != value as ignore=1, so AP is
    computed conditioned on that attribute group (COCOeval honours 'ignore')."""
    d = copy.deepcopy(gt_dict)
    kept = 0
    for ann in d["annotations"]:
        keep = get_attr(ann, key) == value
        ann["ignore"] = 0 if keep else 1
        kept += int(keep)
    return d, kept


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def fp_fn_per_class(coco_gt, dt_list, conf, iou_thr):
    dt_by_img = defaultdict(list)
    for d in dt_list:
        if d.get("score", 1.0) >= conf:
            dt_by_img[d["image_id"]].append(d)
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_item = {"fp": [], "fn": [], "tp": []}
    for img_id in coco_gt.getImgIds():
        gts = coco_gt.imgToAnns.get(img_id, [])
        dts = sorted(dt_by_img.get(img_id, []), key=lambda x: -x["score"])
        matched = set()
        for d in dts:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if g["category_id"] != d["category_id"] or j in matched:
                    continue
                i = iou_xywh(d["bbox"], g["bbox"])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= iou_thr and best_j >= 0:
                matched.add(best_j)
                stats[d["category_id"]]["tp"] += 1
                per_item["tp"].append((img_id, d["category_id"], d["bbox"]))
            else:
                stats[d["category_id"]]["fp"] += 1
                per_item["fp"].append((img_id, d["category_id"], d["bbox"]))
        for j, g in enumerate(gts):
            if j not in matched:
                stats[g["category_id"]]["fn"] += 1
                per_item["fn"].append((img_id, g["category_id"], g["bbox"]))
    return stats, per_item


def dump_crops(per_item, coco_gt, img_root, out_dir, cats, max_crops):
    try:
        from PIL import Image
    except ImportError:
        print("  [crops] Pillow not installed — skipping crop dump")
        return
    imgs = {im["id"]: im for im in coco_gt.dataset["images"]}
    for kind in ("fp", "fn", "tp"):
        n = 0
        for img_id, cid, box in per_item[kind]:
            if n >= max_crops:
                break
            fn = imgs[img_id]["file_name"]
            path = os.path.join(img_root, fn)
            if not os.path.exists(path):
                continue
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            x, y, w, h = box
            pad = 0.25
            box2 = (max(0, x - w * pad), max(0, y - h * pad),
                    x + w * (1 + pad), y + h * (1 + pad))
            crop = im.crop(tuple(map(int, box2)))
            d = os.path.join(out_dir, kind, cats.get(cid, str(cid)))
            os.makedirs(d, exist_ok=True)
            crop.save(os.path.join(d, f"{img_id}_{n}.jpg"))
            n += 1
        print(f"  [crops] {kind}: wrote {n} crops")


def fmt(v):
    return "  n/a" if v != v else f"{v*100:5.1f}"


# ------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="COCO ground-truth json")
    ap.add_argument("--dt", required=True, help="COCO detections json (results list)")
    ap.add_argument("--img-root", default=None, help="image root (needed for --dump-crops)")
    ap.add_argument("--conf", type=float, default=0.30, help="confidence for FP/FN counting")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU match threshold for FP/FN")
    ap.add_argument("--dump-crops", default=None, help="output dir for TP/FP/FN crops")
    ap.add_argument("--max-crops", type=int, default=40, help="max crops per (kind, class)")
    ap.add_argument("--out", default=None, help="write the summary as json here")
    args = ap.parse_args()

    gt_dict = json.load(open(args.gt))
    dt_list = json.load(open(args.dt))
    coco_gt = coco_from_dict(gt_dict)
    coco_dt = coco_gt.loadRes(dt_list)
    cats = {c["id"]: c["name"] for c in gt_dict["categories"]}
    catIds = coco_gt.getCatIds()
    summary = {"by_class": {}, "by_size": {}, "by_attribute": {}, "fp_fn": {}}

    # --- mAP@50 overall + per class ---
    print("\n=== mAP@50 by class ===")
    per_cls = ap50_per_class(coco_gt, coco_dt, "all")
    for cid in catIds:
        print(f"  {cats[cid]:<12} {fmt(per_cls[cid])}")
        summary["by_class"][cats[cid]] = per_cls[cid]
    valid = [v for v in per_cls.values() if v == v]
    mean_map = float(np.mean(valid)) if valid else float("nan")
    print(f"  {'mAP@50':<12} {fmt(mean_map)}")
    summary["by_class"]["_mean"] = mean_map

    # --- by object size ---
    print("\n=== mAP@50 by object size (COCO small/medium/large) ===")
    for area in ("small", "medium", "large"):
        pc = ap50_per_class(coco_gt, coco_dt, area)
        vals = [v for v in pc.values() if v == v]
        m = float(np.mean(vals)) if vals else float("nan")
        print(f"  {area:<8} {fmt(m)}")
        summary["by_size"][area] = m

    # --- by attribute (occlusion / blur) ---
    for key in ATTR_KEYS:
        values = sorted({str(get_attr(a, key)) for a in gt_dict["annotations"]
                         if get_attr(a, key) is not None})
        if not values:
            print(f"\n=== attribute '{key}': not present in annotations — skipped ===")
            continue
        print(f"\n=== mAP@50 by attribute '{key}' ===")
        summary["by_attribute"][key] = {}
        for val in values:
            sub, kept = filter_gt_by_attr(gt_dict, key, type_cast(val, gt_dict, key))
            if kept == 0:
                continue
            pc = ap50_per_class(coco_from_dict(sub), coco_dt, "all")
            vals = [v for v in pc.values() if v == v]
            m = float(np.mean(vals)) if vals else float("nan")
            print(f"  {key}={val:<8} (n={kept:5d})  {fmt(m)}")
            summary["by_attribute"][key][val] = {"mAP50": m, "n": kept}

    # --- FP / FN per class at fixed confidence ---
    print(f"\n=== FP / FN / TP per class  (conf>={args.conf}, IoU>={args.iou}) ===")
    stats, per_item = fp_fn_per_class(coco_gt, dt_list, args.conf, args.iou)
    print(f"  {'class':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'prec':>6} {'rec':>6}")
    for cid in catIds:
        s = stats[cid]
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
        print(f"  {cats[cid]:<12} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} {fmt(prec)} {fmt(rec)}")
        summary["fp_fn"][cats[cid]] = {**s, "precision": prec, "recall": rec}

    # --- crops ---
    if args.dump_crops:
        if not args.img_root:
            print("\n[crops] --img-root is required to dump crops — skipped")
        else:
            print(f"\n=== dumping crops -> {args.dump_crops} ===")
            dump_crops(per_item, coco_gt, args.img_root, args.dump_crops, cats, args.max_crops)

    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\nsummary written to {args.out}")


def type_cast(val_str, gt_dict, key):
    """Cast the string attribute value back to the type stored in the anns."""
    for a in gt_dict["annotations"]:
        v = get_attr(a, key)
        if v is not None and str(v) == val_str:
            return v
    return val_str


if __name__ == "__main__":
    main()

import argparse
import csv
import math
import os


BUZZSET_CLASSES = ("bee", "bumblebee", "hoverfly", "other_insect", "moth")


def _mean_valid(values):
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}"


def _metric_map(rows):
    return {metric: value for metric, value, _notes in rows}


def _summary_text(run_name, split, rows):
    metrics = _metric_map(rows)
    ap = metrics.get("AP50_95", float("nan")) * 100.0
    ap50 = metrics.get("AP50", float("nan")) * 100.0
    recall = metrics.get("recall", float("nan")) * 100.0
    lines = [
        f"Evaluation of {run_name} on {split} is done.",
        f"AP50_95 is {ap:.2f}, AP50 is {ap50:.2f}, and AR@100 is {recall:.2f}.",
        "",
        "Per-class AP50_95:",
    ]
    for name in BUZZSET_CLASSES:
        value = metrics.get(f"{name}_AP50_95", float("nan")) * 100.0
        lines.append(f"- {name}: {value:.2f}")
    return "\n".join(lines)


def evaluate(gt_json, pred_json):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit(
            "pycocotools is required. Run this with the same Python environment "
            "used for YOLOV training/evaluation."
        ) from exc

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    rows = [
        ("AP50_95", stats[0], "COCO AP IoU=0.50:0.95, area=all, maxDets=100"),
        ("AP50", stats[1], "COCO AP IoU=0.50, area=all, maxDets=100"),
        ("AP75", stats[2], "COCO AP IoU=0.75, area=all, maxDets=100"),
        ("AP_small", stats[3], "COCO AP small objects"),
        ("AP_medium", stats[4], "COCO AP medium objects"),
        ("AP_large", stats[5], "COCO AP large objects"),
        ("recall", stats[8], "COCO AR IoU=0.50:0.95, area=all, maxDets=100"),
    ]

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    # precision: T x R x K x A x M. Use area=all, maxDets=100.
    # recall: T x K x A x M. Use area=all, maxDets=100.
    for idx, name in enumerate(BUZZSET_CLASSES):
        if idx >= precision.shape[2]:
            continue
        rows.append(
            (
                f"{name}_AP50_95",
                _mean_valid(precision[:, :, idx, 0, 2]),
                f"Per-class COCO AP for {name}",
            )
        )
        rows.append(
            (
                f"{name}_AP50",
                _mean_valid(precision[0, :, idx, 0, 2]),
                f"Per-class AP50 for {name}",
            )
        )
        rows.append(
            (
                f"{name}_recall",
                _mean_valid(recall[:, idx, 0, 2]),
                f"Per-class AR@100 for {name}",
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract COCO metrics from YOLOV refined prediction files."
    )
    parser.add_argument("--gt", required=True, help="Path to gt_refined.json")
    parser.add_argument("--pred", required=True, help="Path to refined_pred.json")
    parser.add_argument(
        "--out",
        default=os.path.join(
            "buzzset_yolovpp_comparison", "results", "yolovpp_run.csv"
        ),
        help="Output CSV path.",
    )
    parser.add_argument("--run-name", default="yolovpp_swin_tiny")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional text summary path. Defaults to <out> with _summary.txt.",
    )
    args = parser.parse_args()

    rows = evaluate(args.gt, args.pred)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("run", "split", "metric", "value", "notes"))
        for metric, value, notes in rows:
            writer.writerow((args.run_name, args.split, metric, _fmt(value), notes))

    summary_out = args.summary_out
    if summary_out is None:
        base, _ext = os.path.splitext(args.out)
        summary_out = f"{base}_summary.txt"
    summary = _summary_text(args.run_name, args.split, rows)
    with open(summary_out, "w", encoding="utf-8") as handle:
        handle.write(summary)
        handle.write("\n")

    print()
    print(summary)
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()

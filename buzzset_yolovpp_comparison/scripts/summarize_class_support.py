import argparse
import json
from collections import Counter
from pathlib import Path


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(
        description="Per-class instance counts (support) for the BuzzSet splits. "
        "Pair this with the evaluator's per-class AP table: classes with very few "
        "validation instances give noisy AP that drags the unweighted mean down."
    )
    parser.add_argument(
        "--data-root",
        default=str(Path("BuzzSet-org-videos") / "BuzzSetV2_split"),
        help="Folder holding train/valid/test, each with _annotations.coco.json.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument(
        "--low-support",
        type=int,
        default=30,
        help="Flag classes with fewer than this many instances in the eval split.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    first = load(root / args.splits[0] / "_annotations.coco.json")
    cat_name = {c["id"]: c["name"] for c in first["categories"]}
    cat_ids = sorted(cat_name)

    counts = {s: Counter() for s in args.splits}
    for s in args.splits:
        data = load(root / s / "_annotations.coco.json")
        for ann in data["annotations"]:
            counts[s][ann["category_id"]] += 1

    header = f"{'class':14}" + "".join(f"{s:>9}" for s in args.splits) + f"{'total':>9}"
    print(header)
    print("-" * len(header))
    for cid in cat_ids:
        row = [counts[s][cid] for s in args.splits]
        print(f"{cat_name[cid]:14}" + "".join(f"{v:>9}" for v in row) + f"{sum(row):>9}")
    totals = [sum(counts[s].values()) for s in args.splits]
    print("-" * len(header))
    print(f"{'TOTAL':14}" + "".join(f"{v:>9}" for v in totals) + f"{sum(totals):>9}")

    eval_split = "valid" if "valid" in args.splits else args.splits[-1]
    weak = [cat_name[cid] for cid in cat_ids if counts[eval_split][cid] < args.low_support]
    if weak:
        print(
            f"\nLow support in '{eval_split}' (< {args.low_support} instances): "
            + ", ".join(weak)
        )
        print(
            "Their per-class AP is statistically noisy; report it with these counts "
            "and consider a support-weighted summary mAP alongside the plain mean."
        )


if __name__ == "__main__":
    main()

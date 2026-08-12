import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_split(data_root, ann_dir, split, gframe):
    ann_path = ann_dir / f"{split}_yolov_coco.json"
    data = load_json(ann_path)
    split_dir = data_root / split

    images = data["images"]
    anns = data.get("annotations", [])
    image_ids = {image["id"] for image in images}
    category_ids = {cat["id"] for cat in data["categories"]}

    names = [image.get("name") or image.get("file_name") for image in images]
    duplicate_names = [name for name, count in Counter(names).items() if count > 1]
    missing_images = [name for name in names if not (split_dir / name).exists()]
    bad_ann_image_ids = [ann["id"] for ann in anns if ann["image_id"] not in image_ids]
    bad_ann_category_ids = [ann["id"] for ann in anns if ann["category_id"] not in category_ids]
    nonpositive_boxes = []
    for ann in anns:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            nonpositive_boxes.append(ann["id"])

    sequence_lengths = Counter()
    for image in images:
        sequence_lengths[int(image["sid"])] += 1

    by_class = Counter()
    cat_names = {cat["id"]: cat["name"] for cat in data["categories"]}
    for ann in anns:
        by_class[cat_names.get(ann["category_id"], str(ann["category_id"]))] += 1

    lengths = list(sequence_lengths.values())
    retained_frames = sum(length for length in lengths if length >= gframe)
    retained_sequences = sum(1 for length in lengths if length >= gframe)

    return {
        "split": split,
        "annotation": str(ann_path),
        "images": len(images),
        "annotations": len(anns),
        "videos": len(sequence_lengths),
        "missing_images": len(missing_images),
        "duplicate_names": len(duplicate_names),
        "bad_annotation_image_ids": len(bad_ann_image_ids),
        "bad_annotation_category_ids": len(bad_ann_category_ids),
        "nonpositive_boxes": len(nonpositive_boxes),
        "min_video_length": min(lengths) if lengths else 0,
        "max_video_length": max(lengths) if lengths else 0,
        "gframe": gframe,
        "sequences_retained_at_gframe": retained_sequences,
        "frames_retained_at_gframe": retained_frames,
        "class_counts": dict(by_class),
        "examples": {
            "missing_images": missing_images[:10],
            "duplicate_names": duplicate_names[:10],
            "bad_annotation_image_ids": bad_ann_image_ids[:10],
            "bad_annotation_category_ids": bad_ann_category_ids[:10],
            "nonpositive_boxes": nonpositive_boxes[:10],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Audit generated BuzzSet YOLOV annotations.")
    parser.add_argument(
        "--data-root",
        default=str(Path("BuzzSet-org-videos") / "BuzzSetV2_split"),
    )
    parser.add_argument(
        "--ann-dir",
        default=str(Path("buzzset_yolovpp_comparison") / "generated" / "annotations"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--gframe", type=int, default=16)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_dir = Path(args.ann_dir)
    reports = [audit_split(data_root, ann_dir, split, args.gframe) for split in args.splits]

    for report in reports:
        print(
            "{split}: images={images} anns={annotations} videos={videos} "
            "missing={missing_images} dup_names={duplicate_names} "
            "retained@{gframe}={frames_retained_at_gframe}".format(**report)
        )
        if any(
            report[key]
            for key in [
                "missing_images",
                "duplicate_names",
                "bad_annotation_image_ids",
                "bad_annotation_category_ids",
                "nonpositive_boxes",
            ]
        ):
            print(json.dumps(report["examples"], indent=2))

    output_path = ann_dir.parent / "audit_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()


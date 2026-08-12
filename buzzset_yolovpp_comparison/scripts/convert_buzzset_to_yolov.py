import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_RE = re.compile(
    r"^(?P<key>\d{8}_plot[^_]+_\d{2}\.\d{2})_(?P<frame>\d+)\.[^.]+$"
)


def infer_video_key_and_frame(file_name, fallback_frame=0):
    base = Path(file_name).name
    match = VIDEO_RE.match(base)
    if match:
        return match.group("key"), int(match.group("frame"))

    stem = Path(base).stem
    if "_" in stem:
        key, maybe_frame = stem.rsplit("_", 1)
        if maybe_frame.isdigit():
            return key, int(maybe_frame)
    return stem, int(fallback_frame)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def convert_split(source_root, output_dir, split):
    split_dir = source_root / split
    ann_path = split_dir / "_annotations.coco.json"
    source = load_json(ann_path)

    grouped = defaultdict(list)
    for image in source["images"]:
        key, frame = infer_video_key_and_frame(image["file_name"], image["id"])
        grouped[key].append((frame, image))

    videos = []
    images = []
    old_to_new_image_id = {}
    next_image_id = 1

    for sid, key in enumerate(sorted(grouped)):
        ordered = sorted(grouped[key], key=lambda item: (item[0], item[1]["file_name"]))
        file_names = [image["file_name"] for _, image in ordered]
        widths = Counter([image["width"] for _, image in ordered])
        heights = Counter([image["height"] for _, image in ordered])
        videos.append(
            {
                "id": sid,
                "name": key,
                "file_names": file_names,
                "width": widths.most_common(1)[0][0],
                "height": heights.most_common(1)[0][0],
                "length": len(file_names),
            }
        )

        for fid, (frame_index, image) in enumerate(ordered):
            new_image = dict(image)
            old_to_new_image_id[image["id"]] = next_image_id
            new_image["id"] = next_image_id
            new_image["sid"] = sid
            new_image["fid"] = fid
            new_image["frame_index"] = frame_index
            new_image["name"] = image["file_name"]
            new_image["video_key"] = key
            images.append(new_image)
            next_image_id += 1

    annotations = []
    for ann_id, ann in enumerate(source.get("annotations", []), start=1):
        image_id = ann["image_id"]
        if image_id not in old_to_new_image_id:
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = old_to_new_image_id[image_id]
        bbox = new_ann.get("bbox", [0, 0, 0, 0])
        new_ann["area"] = float(new_ann.get("area", bbox[2] * bbox[3]))
        new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
        new_ann.setdefault("segmentation", [])
        annotations.append(new_ann)

    converted = {
        "info": {
            "description": f"BuzzSet {split} converted to YOLOV video-style COCO",
            "source_annotation": str(ann_path),
        },
        "licenses": source.get("licenses", []),
        "categories": source["categories"],
        "videos": videos,
        "images": images,
        "annotations": annotations,
    }

    output_path = output_dir / "annotations" / f"{split}_yolov_coco.json"
    dump_json(output_path, converted)

    missing_images = [
        image["file_name"] for image in source["images"] if not (split_dir / image["file_name"]).exists()
    ]
    class_by_id = {cat["id"]: cat["name"] for cat in source["categories"]}
    class_counts = Counter(class_by_id.get(ann["category_id"], ann["category_id"]) for ann in annotations)
    lengths = [video["length"] for video in videos]
    summary = {
        "split": split,
        "source_annotation": str(ann_path),
        "output_annotation": str(output_path),
        "images": len(images),
        "annotations": len(annotations),
        "videos": len(videos),
        "missing_images": len(missing_images),
        "video_length_min": min(lengths) if lengths else 0,
        "video_length_max": max(lengths) if lengths else 0,
        "video_length_ge_8": sum(1 for length in lengths if length >= 8),
        "video_length_ge_16": sum(1 for length in lengths if length >= 16),
        "video_length_ge_32": sum(1 for length in lengths if length >= 32),
        "frames_in_videos_ge_16": sum(length for length in lengths if length >= 16),
        "frames_in_videos_ge_32": sum(length for length in lengths if length >= 32),
        "class_counts": dict(class_counts),
        "missing_image_examples": missing_images[:10],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert BuzzSet COCO splits to YOLOV video-style COCO.")
    parser.add_argument(
        "--source-root",
        default=str(Path("BuzzSet-org-videos") / "BuzzSetV2_split"),
        help="Directory containing train/valid/test BuzzSet splits.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("buzzset_yolovpp_comparison") / "generated"),
        help="Directory where generated annotations and summaries are written.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    summaries = [convert_split(source_root, output_dir, split) for split in args.splits]
    dump_json(output_dir / "conversion_summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()


import argparse
import csv
from pathlib import Path


def read_metrics(path):
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "metric", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            split = row["split"].strip()
            metric = row["metric"].strip()
            value = row["value"].strip()
            if not split or not metric or value == "":
                continue
            rows[(split, metric)] = {
                "value": float(value),
                "run": row.get("run", Path(path).stem),
                "notes": row.get("notes", ""),
            }
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and YOLOV++ metric CSV files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    baseline = read_metrics(args.baseline)
    candidate = read_metrics(args.candidate)
    keys = sorted(set(baseline) | set(candidate))

    print("split,metric,baseline,candidate,delta")
    for key in keys:
        base = baseline.get(key)
        cand = candidate.get(key)
        base_value = "" if base is None else f"{base['value']:.6g}"
        cand_value = "" if cand is None else f"{cand['value']:.6g}"
        delta = "" if base is None or cand is None else f"{cand['value'] - base['value']:.6g}"
        print(f"{key[0]},{key[1]},{base_value},{cand_value},{delta}")


if __name__ == "__main__":
    main()
